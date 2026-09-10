"""
D10: regression guards for validation paths that were only ever checked ad
hoc during D1/D2/D4 (RotationShape overrun, hamiltonian_coeffs length,
unknown circuit_type) -- never persisted as unit tests until now.
Author: Aritra Bal (ETP)
2026-09-05
"""
import unittest

import pennylane as qml
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

    def test_missing_key_raises(self) -> None:
        circuit = VQCCircuit(num_layers=1)
        wires = list(range(4))
        weights = type('W', (), {'aux': {}})()
        with self.assertRaises(KeyError):
            circuit.measure(weights, wires)

    def test_coeffs_train_independently_per_wire(self) -> None:
        """Each wire's coefficient must get its own gradient (D2 extension)."""
        circuit = VQCCircuit(num_layers=1)
        wires = list(range(2))
        dev = qml.device('default.qubit', wires=len(wires))

        @qml.qnode(dev)
        def score(coeffs):
            qml.RY(0.3, wires=0)
            qml.RY(0.7, wires=1)
            weights = type('W', (), {'aux': {'hamiltonian_coeffs': coeffs}})()
            return circuit.measure(weights, wires)

        coeffs = np.array([0.1, 0.1], requires_grad=True)
        gradient = qml.grad(score)(coeffs)
        expected = np.array([np.cos(0.3), np.cos(0.7)])
        self.assertTrue(np.allclose(gradient, expected))
        self.assertNotAlmostEqual(float(gradient[0]), float(gradient[1]))


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
