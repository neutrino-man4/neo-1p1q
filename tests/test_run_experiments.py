"""
Verify repeated-run parsing, seed generation, and bounded process scheduling.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

import run_experiments
from helpers.config import DEFAULT_CONFIG


class _FakeProcess:
    """Small controllable Popen replacement for scheduler tests."""

    active = 0
    maximum_active = 0

    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.polls = 0
        self.signals = []
        type(self).active += 1
        type(self).maximum_active = max(type(self).maximum_active, type(self).active)

    def poll(self):
        self.polls += 1
        if self.polls == 1:
            return None
        type(self).active -= 1
        return self.return_code

    def send_signal(self, sent_signal):
        self.signals.append(sent_signal)

    def wait(self):
        if self.polls < 2:
            type(self).active -= 1
        self.polls = 2
        return self.return_code


class TestRunExperiments(unittest.TestCase):
    """The launcher must keep orchestration outside each training config."""

    def setUp(self) -> None:
        _FakeProcess.active = 0
        _FakeProcess.maximum_active = 0

    def _config(self, directory: str, **updates) -> str:
        cfg = OmegaConf.merge(
            OmegaConf.load(DEFAULT_CONFIG),
            {'save_dir': str(Path(directory) / 'models'), 'seed': 'experiment'},
            updates,
        )
        path = Path(directory) / 'input.yaml'
        OmegaConf.save(cfg, path)
        return str(path)

    def test_main_freezes_config_and_generates_random_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory)

            def inspect_launch(path, seeds, cores):
                frozen = OmegaConf.load(path)
                self.assertEqual(frozen.seed, 'changed')
                self.assertNotIn('number_of_runs', frozen)
                self.assertNotIn('num_cores', frozen)
                self.assertEqual(len(seeds), 3)
                self.assertEqual(len(set(seeds)), 3)
                self.assertTrue(all(isinstance(seed, int) and seed >= 0 for seed in seeds))
                self.assertEqual(cores, 2)
                return 0

            with patch.object(run_experiments, 'launch_runs', side_effect=inspect_launch):
                result = run_experiments.main([
                    '--config', config_path,
                    '--number-of-runs', '3',
                    '--num-cores', '2',
                    'seed=changed',
                ])
            self.assertEqual(result, 0)

    def test_main_rejects_random_seed_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory)
            with self.assertRaisesRegex(ValueError, 'generated per run'):
                run_experiments.main(['--config', config_path, 'random_seed=7'])

    def test_launcher_rejects_resume_and_misplaced_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resume_path = self._config(directory, resume=True)
            with self.assertRaisesRegex(ValueError, 'fresh runs'):
                run_experiments.main(['--config', resume_path])

            misplaced_path = self._config(directory, number_of_runs=2)
            with self.assertRaisesRegex(ValueError, 'launcher options'):
                run_experiments.main(['--config', misplaced_path])

    def test_existing_destination_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory)
            destination = Path(directory) / 'models' / 'experiment' / '42'
            destination.mkdir(parents=True)
            with patch.object(run_experiments, '_generate_random_seeds', return_value=[42]), \
                    patch.object(run_experiments, 'launch_runs') as launch:
                with self.assertRaisesRegex(FileExistsError, str(destination)):
                    run_experiments.main(['--config', config_path])
            launch.assert_not_called()

    def test_scheduler_limits_active_processes(self) -> None:
        started = []

        def start_process(_path, seed):
            started.append(seed)
            return _FakeProcess()

        with patch.object(run_experiments, '_start_training', side_effect=start_process), \
                patch.object(run_experiments.time, 'sleep'):
            result = run_experiments.launch_runs('config.yaml', [10, 11, 12, 13], 2)

        self.assertEqual(result, 0)
        self.assertEqual(started, [10, 11, 12, 13])
        self.assertEqual(_FakeProcess.maximum_active, 2)

    def test_scheduler_stops_after_failure(self) -> None:
        started = []

        def start_process(_path, seed):
            started.append(seed)
            return _FakeProcess(return_code=1 if seed == 10 else 0)

        with patch.object(run_experiments, '_start_training', side_effect=start_process), \
                patch.object(run_experiments.time, 'sleep'):
            result = run_experiments.launch_runs('config.yaml', [10, 11, 12], 1)

        self.assertEqual(result, 1)
        self.assertEqual(started, [10])

    def test_child_uses_requested_seed_and_one_thread(self) -> None:
        captured = {}

        def popen(command, **kwargs):
            captured['command'] = command
            captured.update(kwargs)
            return _FakeProcess()

        with patch.object(run_experiments.subprocess, 'Popen', side_effect=popen):
            run_experiments._start_training('/tmp/config.yaml', 19)

        self.assertEqual(captured['command'][-1], 'random_seed=19')
        self.assertEqual(captured['command'][-3:-1], ['--config', '/tmp/config.yaml'])
        self.assertTrue(captured['start_new_session'])
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            self.assertEqual(captured['env'][name], '1')
        self.assertEqual(captured['env'].get('PATH'), os.environ.get('PATH'))

    def test_keyboard_interrupt_stops_active_children(self) -> None:
        process = _FakeProcess()
        calls = 0

        def interrupt_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return None

        process.poll = interrupt_once
        with patch.object(run_experiments, '_start_training', return_value=process):
            result = run_experiments.launch_runs('config.yaml', [10], 1)

        self.assertEqual(result, 130)
        self.assertEqual(process.signals, [signal.SIGINT])


if __name__ == '__main__':
    unittest.main()
