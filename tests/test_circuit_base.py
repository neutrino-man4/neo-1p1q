"""
D10: regression guards for validation paths that were only ever checked ad
hoc during D1/D2/D4 (RotationShape overrun, hamiltonian_coeffs length,
unknown circuit_type) -- never persisted as unit tests until now.
Author: Aritra Bal (ETP)
2026-09-05
"""
import unittest

import pennylane.numpy as np

from quantum.circuits.base import RotationShape
from quantum.circuits.vqc import VQCCircuit
from quantum.circuits import registry


class TestRotationShapeValidate(unittest.TestCase):
    """RotationShape.validate() must reject layer/wire overruns (D1)."""

    def setUp(self) -> None:
        self.shape = RotationShape(L=2, N=4, R=3)

    def test_layer_overrun_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.shape.validate(layer=2, wires=list(range(4)))

    def test_wire_overrun_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.shape.validate(layer=0, wires=list(range(5)))

    def test_in_bounds_does_not_raise(self) -> None:
        self.shape.validate(layer=1, wires=list(range(4)))


class TestVQCMeasureValidation(unittest.TestCase):
    """VQCCircuit.measure() must reject a hamiltonian_coeffs/wires length mismatch (D2)."""

    def test_wrong_length_coeffs_raises(self) -> None:
        circuit = VQCCircuit(num_layers=1)
        wires = list(range(4))
        weights = type('W', (), {'aux': {'hamiltonian_coeffs': [0.1, 0.1]}})()
        with self.assertRaises(ValueError):
            circuit.measure(weights, wires)

    def test_default_length_matches_wires(self) -> None:
        circuit = VQCCircuit(num_layers=1)
        wires = list(range(4))
        weights = type('W', (), {'aux': {}})()
        # should not raise -- default hamiltonian_coeffs is [0.1] * len(wires)
        circuit.measure(weights, wires)


class TestRegistryUnknownCircuit(unittest.TestCase):
    """registry.get() must raise ValueError, listing registered names, for an unknown type (D4)."""

    def test_unknown_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            registry.get('CNN', num_layers=1)

    def test_known_type_returns_instance(self) -> None:
        circuit = registry.get('normal', num_layers=1)
        self.assertIsInstance(circuit, VQCCircuit)


if __name__ == '__main__':
    unittest.main()
