"""
helpers.config replaces Hydra: load a YAML, apply key=value CLI overrides, and
save the exact merged config to one reusable file. These tests pin the round
trip that matters -- a run's config.yaml must reload identically and be usable
as the --config for a later run.
Author: Aritra Bal (ETP)
2026-09-05
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

from omegaconf import OmegaConf

from helpers.config import DEFAULT_CONFIG, load_config, save_config

_BASE = DEFAULT_CONFIG


class TestLoadConfig(unittest.TestCase):
    """load_config() must apply typed + nested overrides and honour --print-config."""

    def _run(self, argv):
        with mock.patch.object(sys, 'argv', ['prog'] + argv):
            return load_config(default_config=_BASE)

    def test_no_overrides_returns_base(self):
        cfg = self._run([])
        base = OmegaConf.load(_BASE)
        self.assertEqual(cfg, base)

    def test_typed_and_nested_overrides(self):
        cfg = self._run(['epochs=7', 'save=false', 'aux_weights.scale_factor=2.5', 'n_signal=40'])
        self.assertEqual(cfg.epochs, 7)
        self.assertIsInstance(cfg.epochs, int)
        self.assertIs(cfg.save, False)
        self.assertEqual(cfg.aux_weights.scale_factor, 2.5)
        self.assertEqual(cfg.n_signal, 40)
        self.assertEqual(cfg.signal, 'TTBar_')  # untouched base key preserved

    def test_print_config_exits(self):
        with self.assertRaises(SystemExit):
            self._run(['--print-config'])

    def test_explicit_config_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'custom.yaml')
            OmegaConf.save(OmegaConf.create({'wires': 9, 'seed': 'x'}), path)
            with mock.patch.object(sys, 'argv', ['prog', '--config', path, 'seed=y']):
                cfg = load_config()
            self.assertEqual(cfg.wires, 9)
            self.assertEqual(cfg.seed, 'y')


class TestSaveConfigRoundTrip(unittest.TestCase):
    """save_config -> OmegaConf.load must reproduce the run config exactly."""

    def test_exact_single_file_round_trip(self):
        base = OmegaConf.load(_BASE)
        cfg = OmegaConf.merge(
            base,
            OmegaConf.from_dotlist(['seed=exp042', 'epochs=7', 'device_name=default.qubit',
                                    'aux_weights.bias=0.2', 'n_signal=40']),
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'config.yaml')
            save_config(cfg, path)
            reloaded = OmegaConf.load(path)

            self.assertEqual(reloaded, cfg)
            self.assertEqual(reloaded.seed, 'exp042')
            self.assertEqual(reloaded.aux_weights.bias, 0.2)

            # byte-stable: saving the reloaded config again is identical
            path2 = os.path.join(d, 'config2.yaml')
            save_config(reloaded, path2)
            with open(path) as a, open(path2) as b:
                self.assertEqual(a.read(), b.read())

    def test_saved_config_is_reusable_as_input(self):
        """A saved run config must load straight back through load_config(--config ...)."""
        cfg = OmegaConf.merge(OmegaConf.load(_BASE), OmegaConf.from_dotlist(['seed=r1', 'epochs=3']))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'config.yaml')
            save_config(cfg, path)
            with mock.patch.object(sys, 'argv', ['prog', '--config', path]):
                reused = load_config()
            self.assertEqual(reused, cfg)


if __name__ == '__main__':
    unittest.main()
