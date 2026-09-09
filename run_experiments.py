"""
Launch repeated training runs with consecutive random seeds and bounded parallelism.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Sequence

from omegaconf import OmegaConf

from helpers.config import (
    DEFAULT_CONFIG,
    SINGLE_THREAD_ENV,
    load_config_values,
    positive_integer,
    run_directory,
    save_config,
    validate_training_config,
)

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launcher settings separately from training overrides."""
    parser = argparse.ArgumentParser(
        description='Launch repeated training runs with consecutive random seeds.'
    )
    parser.add_argument(
        '--config', default=DEFAULT_CONFIG,
        help=f'path to the training YAML (default: {DEFAULT_CONFIG})',
    )
    parser.add_argument('--number-of-runs', type=positive_integer, default=1)
    parser.add_argument('--num-cores', type=positive_integer, default=1)
    parser.add_argument(
        'overrides', nargs='*',
        help='training overrides, e.g. seed=experiment epochs=5 random_seed=42',
    )
    return parser.parse_args(argv)


def _start_training(config_path: str, random_seed: int) -> subprocess.Popen:
    """Start one single-threaded training subprocess."""
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name('train.py')),
        '--config',
        config_path,
        f'random_seed={random_seed}',
    ]
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENV)
    print(f'Launching random_seed={random_seed}')
    return subprocess.Popen(command, env=environment, start_new_session=True)


def launch_runs(config_path: str, random_seeds: Sequence[int], num_cores: int) -> int:
    """Run at most ``num_cores`` single-threaded training processes concurrently."""
    pending = iter(random_seeds)
    active: dict[subprocess.Popen, int] = {}
    failed = False
    exhausted = False

    try:
        while active or not exhausted:
            while not failed and not exhausted and len(active) < num_cores:
                try:
                    random_seed = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                try:
                    process = _start_training(config_path, random_seed)
                except OSError as error:
                    failed = True
                    print(
                        f'Could not launch random_seed={random_seed}: {error}',
                        file=sys.stderr,
                    )
                    break
                active[process] = random_seed

            if not active:
                break

            completed: list[tuple[subprocess.Popen, int]] = []
            while not completed:
                for process in active:
                    return_code = process.poll()
                    if return_code is not None:
                        completed.append((process, return_code))
                if not completed:
                    time.sleep(0.1)

            for process, return_code in completed:
                random_seed = active.pop(process)
                if return_code == 0:
                    print(f'Completed random_seed={random_seed}')
                else:
                    failed = True
                    print(
                        f'Training failed for random_seed={random_seed} '
                        f'with exit code {return_code}.',
                        file=sys.stderr,
                    )
    except KeyboardInterrupt:
        for process in active:
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
        for process in active:
            process.wait()
        return 130

    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve one experiment and launch its requested independent runs."""
    args = _parse_args(argv)
    cfg = load_config_values(args.config, args.overrides)
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    random_seed = validate_training_config(cfg, require_random_seed=True)
    if cfg.get('resume', False):
        raise ValueError('run_experiments.py only launches fresh runs; resume runs individually.')

    last_random_seed = random_seed + args.number_of_runs - 1
    random_seeds = list(range(random_seed, last_random_seed + 1))

    legacy_config = run_directory(cfg.save_dir, cfg.seed) / 'config.yaml'
    if legacy_config.exists():
        raise FileExistsError(
            f'Legacy run already uses experiment directory {legacy_config.parent}.'
        )
    destinations = [run_directory(cfg.save_dir, cfg.seed, seed) for seed in random_seeds]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError('Run directories already exist: ' + ', '.join(existing))

    with tempfile.TemporaryDirectory(prefix='neo1p1q-runs-') as directory:
        resolved_config = str(Path(directory) / 'config.yaml')
        save_config(cfg, resolved_config)
        return launch_runs(resolved_config, random_seeds, args.num_cores)


if __name__ == '__main__':
    raise SystemExit(main())
