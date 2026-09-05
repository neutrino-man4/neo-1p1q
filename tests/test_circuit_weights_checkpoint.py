"""
D7 checkpoint round-trip: a CircuitWeights instance (rot array + named aux
scalars, both trainable) must survive helpers.utils.Pickle/Unpickle exactly
-- the same save/load functions QuantumTrainer.save() and
QuantumClassifier.load_weights() already use.
Author: Aritra Bal (ETP)
2026-09-05
"""
import os
import unittest

import numpy as onp
import pennylane.numpy as np

import helpers.utils as ut
from quantum.circuits.base import CircuitWeights


class TestCircuitWeightsCheckpoint(unittest.TestCase):
    """CircuitWeights must round-trip losslessly through the existing pickle helpers."""

    def setUp(self) -> None:
        onp.random.seed(0)
        self.weights = CircuitWeights(
            rot=np.array(onp.random.uniform(0, np.pi, size=(2, 3, 3)), requires_grad=True),
            aux={
                'scale_factor': np.array(1.0, requires_grad=True),
                'bias': np.array(0.1, requires_grad=True),
                'hamiltonian_coeffs': np.array([0.1] * 3, requires_grad=True),
            },
        )
        self.filename = 'test_circuit_weights_checkpoint.pickle'

    def tearDown(self) -> None:
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_round_trip(self) -> None:
        """Save then load must reproduce rot and every aux value exactly."""
        ut.Pickle({'weights': self.weights}, self.filename)
        loaded = ut.Unpickle(self.filename)['weights']

        self.assertIsInstance(loaded, CircuitWeights)
        self.assertTrue(onp.allclose(loaded.rot, self.weights.rot))
        for key, value in self.weights.aux.items():
            self.assertTrue(onp.allclose(loaded.aux[key], value))


if __name__ == '__main__':
    unittest.main()
