"""
D9 wiring test: QuantumClassifier.set_circuit()'s registry-based circuit and
state-circuit must match VQCCircuit/_state_build called directly. This test
used to compare the new implementation against the pre-switchover
_vqc_circuit/_vqc_state_circuit methods (D8); those methods are deleted as of
D9, so it now guards the wiring in set_circuit() instead -- the circuit
body's correctness against the old implementation was already locked down by
D8's original comparison, recorded in refactored.MD.
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
            aux={'scale_factor': np.array(1.0), 'bias': np.array(0.1)},
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
        self.assertAlmostEqual(float(wired_out), float(standalone_out), places=10)

    def test_state_matches(self) -> None:
        """qc.state_circuit_qnode must match a hand-built state QNode for the same circuit."""
        standalone = qml.QNode(
            lambda w, i: VQCCircuit(num_layers=self.num_layers).build(
                w, i, self.wires, measure_override=lambda *_: qml.state()
            ),
            self.qc.device,
            interface=self.qc.backend,
        )
        wired_state = onp.array(self.qc.state_circuit_qnode(self.weights, self.inputs))
        standalone_state = onp.array(standalone(self.weights, self.inputs))
        self.assertTrue(onp.allclose(wired_state, standalone_state, atol=1e-10))


if __name__ == '__main__':
    unittest.main()
