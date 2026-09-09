"""
Minimal config handling for train.py / evaluate.py.

The repo only ever needed "load one YAML, allow key=value overrides on the CLI,
and save the exact config a run used to a single reusable file". That is done
here with OmegaConf directly -- no Hydra. `cfg` stays a DictConfig, so the dot
and bracket access the rest of the code relies on is unchanged.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""
import argparse
import os

from omegaconf import DictConfig, OmegaConf

# Anchored to the repo root (parent of helpers/) so it resolves regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "VQC", "base.yaml")


def load_config(default_config: str = DEFAULT_CONFIG) -> DictConfig:
    """
    Parse the CLI and return the merged config.

    Usage: ``python train.py [--config PATH] [--resume] [--print-config] [key=value ...]``
    Overrides use OmegaConf dotlist syntax, e.g. ``epochs=5 aux_weights.bias=0.2``.
    A saved run config (``<save_dir>/<seed>/config.yaml``) can be passed straight
    back in via ``--config`` to reproduce that run.

    Args:
        default_config: base YAML used when ``--config`` is not given.

    Returns:
        The base config merged with any CLI overrides.
    """
    parser = argparse.ArgumentParser(description="Load a YAML config with CLI overrides.")
    parser.add_argument(
        "--config", default=default_config,
        help="path to the base YAML config (default: " + default_config + ")",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="print the resolved config and exit",
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='resume from the latest checkpoint for the selected run',
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf overrides, e.g. seed=run1 epochs=5 aux_weights.bias=0.2",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.resume:
        cfg.resume = True

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        raise SystemExit(0)

    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    """Write the exact config used for a run to a single reusable YAML file."""
    OmegaConf.save(cfg, path, resolve=True)


def evaluation_config_path(argv: list[str] | None = None) -> str:
    """Locate a saved run YAML without accepting training-option overrides."""
    parser = argparse.ArgumentParser(description="Evaluate a saved training run.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help='Path to the saved run config.yaml')
    source.add_argument('--seed', help='Run name under the model directory')
    parser.add_argument(
        '--model-dir', default=None,
        help='Model directory (default: save_dir from configs/VQC/base.yaml)',
    )
    args = parser.parse_args(argv)
    if args.config:
        if args.model_dir is not None:
            parser.error('--model-dir can only be used with --seed')
        return os.path.abspath(os.path.expanduser(args.config))
    if args.seed in ('.', '..') or os.path.basename(args.seed) != args.seed:
        parser.error('--seed must be a run directory name')
    model_dir = args.model_dir or OmegaConf.load(DEFAULT_CONFIG).save_dir
    return os.path.abspath(os.path.join(os.path.expanduser(model_dir), args.seed, 'config.yaml'))
