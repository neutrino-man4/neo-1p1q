"""
D9 wiring test: QuantumClassifier.set_circuit()'s registry-based circuit must
match VQCCircuit called directly. This test used to compare the new
implementation against the pre-switchover _vqc_circuit method (D8); that
method is deleted as of D9, so it now guards the wiring in set_circuit()
instead -- the circuit body's correctness against the old implementation was
already locked down by D8's original comparison, recorded in refactored.MD.
Author: Aritra Bal (ETP)
2026-09-05
"""
import unittest

import numpy as onp
import pennylane as qml
import pennylane.numpy as np

import quantum.architectures as arch
from quantum.circuits.base import CircuitWeights
from quantum.circuits.vqc import VQCCircuit


class TestVQCCircuitWiring(unittest.TestCase):
    """set_circuit()'s wired-up circuit must match a standalone VQCCircuit."""

    def setUp(self) -> None:
        onp.random.seed(0)
        self.num_layers = 2
        self.n_qubits = 4
        self.wires = list(range(self.n_qubits))
        self.weights = CircuitWeights(
            rot=np.array(
                onp.random.uniform(0, np.pi, size=(self.num_layers, self.n_qubits, 3))
            ),
            aux={
                'scale_factor': np.array(1.0),
                'bias': np.array(0.1),
                'hamiltonian_coeffs': np.array([0.1] * self.n_qubits),
            },
        )
        self.inputs = np.array(
            onp.random.uniform(-1, 1, size=(1, self.n_qubits * self.num_layers, 3))
        )
        self.qc = arch.QuantumClassifier(
            wires=self.n_qubits,
            layers=self.num_layers,
            shots=None,
            dev_name='default.qubit',
            backend_name='autograd',
        )
        self.qc.set_circuit('normal')

    def test_expval_matches(self) -> None:
        """qc.circuit (built by set_circuit) must match VQCCircuit.build() directly."""
        standalone = qml.QNode(
            VQCCircuit(num_layers=self.num_layers).build, self.qc.device, interface=self.qc.backend
        )
        wired_out = self.qc.circuit(self.weights, self.inputs)
        standalone_out = standalone(self.weights, self.inputs, self.wires)
        self.assertEqual(wired_out.shape, (1,))
        self.assertEqual(standalone_out.shape, (1,))
        self.assertAlmostEqual(float(wired_out[0]), float(standalone_out[0]), places=10)

    def test_expval_golden_value(self) -> None:
        """
        Pin the exact analytic expval for this fixed seed/weights/input. Guards
        against cross-version drift in PennyLane's gate math -- default.qubit
        with shots=None is exact linear algebra, so this constant must hold on
        any PennyLane version. Reference computed on PennyLane 0.37.0.
        """
        out = float(self.qc.circuit(self.weights, self.inputs)[0])
        self.assertAlmostEqual(out, 0.034755910654483246, places=12)


class TestSetCircuitDiffMethodValidation(unittest.TestCase):
    """set_circuit() must reject an incompatible diff_method with a clear, actionable error."""

    def test_backprop_on_finite_shot_lightning_qubit_raises_with_suggestion(self) -> None:
        qc = arch.QuantumClassifier(
            wires=2, layers=1, shots=50, dev_name='lightning.qubit', backend_name='autograd',
        )
        with self.assertRaises(ValueError) as context:
            qc.set_circuit('normal', diff_method='backprop')
        message = str(context.exception)
        self.assertIn('backprop', message)
        self.assertIn('parameter-shift', message)

    def test_parameter_shift_on_finite_shot_lightning_qubit_succeeds(self) -> None:
        qc = arch.QuantumClassifier(
            wires=2, layers=1, shots=50, dev_name='lightning.qubit', backend_name='autograd',
        )
        qc.set_circuit('normal', diff_method='parameter-shift')
        self.assertIsNotNone(qc.circuit)


if __name__ == '__main__':
    unittest.main()
