"""
Verify parallel experiment evaluation, incomplete-run handling, and ROC summaries.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import contextlib
import io
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate
from helpers.config import parse_evaluation_args
from helpers.trained_run import CIRCUIT_FILES


class _FakeProcess:
    """Controllable subprocess replacement for scheduler tests."""

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


class TestEvaluationExperimentCLI(unittest.TestCase):
    """Experiment mode must remain separate from single-run selection."""

    def test_experiment_directory_and_core_count(self) -> None:
        args = parse_evaluation_args([
            '--experiment-dir', '/tmp/experiment', '--num-cores', '3',
        ])
        self.assertEqual(args.experiment_dir, '/tmp/experiment')
        self.assertEqual(args.num_cores, 3)

    def test_rejects_ambiguous_or_invalid_options(self) -> None:
        invalid = [
            ['--experiment-dir', '/tmp/run', '--random-seed', '42'],
            ['--experiment-dir', '/tmp/run', '--model-dir', '/tmp/models'],
            ['--experiment-dir', '/tmp/run', '--num-cores', '0'],
            ['--config', '/tmp/config.yaml', '--num-cores', '2'],
        ]
        for args in invalid:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_evaluation_args(args)


class TestExperimentEvaluation(unittest.TestCase):
    """Incomplete or failed runs must not stop other evaluations."""

    def setUp(self) -> None:
        _FakeProcess.active = 0
        _FakeProcess.maximum_active = 0

    def _complete_run(self, root: Path, random_seed: int) -> Path:
        run_dir = root / str(random_seed)
        circuit_dir = run_dir / 'circuits'
        circuit_dir.mkdir(parents=True)
        (run_dir / 'config.yaml').touch()
        (run_dir / 'trained_model.pickle').touch()
        for name in CIRCUIT_FILES:
            (circuit_dir / name).touch()
        return run_dir / 'config.yaml'

    def test_discovers_numeric_runs_and_records_incomplete_ones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = self._complete_run(root, 42)
            (root / '43').mkdir()
            (root / 'notes').mkdir()

            total, runnable, failures, notices = evaluate.discover_evaluation_runs(root)

        self.assertEqual(total, 2)
        self.assertEqual(runnable, [complete])
        self.assertIn('43', failures)
        self.assertIn('config.yaml', failures['43'])
        self.assertEqual(len(notices), 1)
        self.assertIn('notes', notices[0])

    def test_scheduler_limits_processes_and_continues_after_failure(self) -> None:
        paths = [Path(f'/tmp/{seed}/config.yaml') for seed in (42, 43, 44)]
        started = []

        def start(config_path):
            random_seed = int(config_path.parent.name)
            started.append(random_seed)
            return _FakeProcess(return_code=1 if random_seed == 42 else 0)

        with patch.object(evaluate, '_start_evaluation', side_effect=start), \
                patch.object(evaluate.time, 'sleep'):
            successful, failures, interrupted = evaluate.evaluate_configs(paths, 2)

        self.assertFalse(interrupted)
        self.assertEqual(started, [42, 43, 44])
        self.assertEqual([path.parent.name for path in successful], ['43', '44'])
        self.assertIn('42', failures)
        self.assertEqual(_FakeProcess.maximum_active, 2)

    def test_child_uses_one_computational_thread(self) -> None:
        captured = {}

        def popen(command, **kwargs):
            captured['command'] = command
            captured.update(kwargs)
            return _FakeProcess()

        path = Path('/tmp/42/config.yaml')
        with patch.object(evaluate.subprocess, 'Popen', side_effect=popen):
            evaluate._start_evaluation(path)

        self.assertEqual(captured['command'][-2:], ['--config', str(path)])
        self.assertTrue(captured['start_new_session'])
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            self.assertEqual(captured['env'][name], '1')
        self.assertEqual(captured['env'].get('PATH'), os.environ.get('PATH'))

    def test_keyboard_interrupt_stops_active_evaluations(self) -> None:
        process = _FakeProcess()
        process.poll = lambda: (_ for _ in ()).throw(KeyboardInterrupt)
        path = Path('/tmp/42/config.yaml')
        with patch.object(evaluate, '_start_evaluation', return_value=process):
            successful, failures, interrupted = evaluate.evaluate_configs([path], 1)

        self.assertTrue(interrupted)
        self.assertEqual(successful, [])
        self.assertEqual(failures, {})
        self.assertEqual(process.signals, [signal.SIGINT])

    def test_summary_reports_auc_and_writes_shaded_roc(self) -> None:
        results = [
            {
                'labels': np.array([0, 0, 1, 1]),
                'scores': np.array([0.1, 0.2, 0.8, 0.9]),
                'auc': 1.0,
            },
            {
                'labels': np.array([0, 0, 1, 1]),
                'scores': np.array([0.1, 0.8, 0.7, 0.9]),
                'auc': 0.75,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = evaluate.summarize_evaluations(root, results, total_runs=3)
            plot_exists = (root / 'roc_curve_summary.png').is_file()

        self.assertIn('Runs in total: 3', report)
        self.assertIn('Runs evaluated successfully: 2', report)
        self.assertIn('Jets evaluated per run: 4', report)
        self.assertIn('Total jet evaluations: 8', report)
        self.assertIn('AUC: 0.8750 +/- 0.1250', report)
        self.assertTrue(plot_exists)

    def test_incomplete_run_is_logged_without_blocking_summary(self) -> None:
        result = {
            'labels': np.array([0, 0, 1, 1]),
            'scores': np.array([0.1, 0.2, 0.8, 0.9]),
            'auc': 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._complete_run(root, 42)
            (root / '43').mkdir()
            with patch.object(
                evaluate, 'evaluate_configs',
                return_value=([config_path], {}, False),
            ), patch.object(evaluate, '_load_result', return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                return_code = evaluate.evaluate_experiment(str(root), num_cores=2)
            log = (root / 'evaluation_summary.log').read_text()

        self.assertEqual(return_code, 0)
        self.assertIn('random_seed=43: missing config.yaml', log)
        self.assertIn('Runs in total: 2', log)
        self.assertIn('Runs evaluated successfully: 1', log)


if __name__ == '__main__':
    unittest.main()
