"""
Parity test: VQCCircuit (D2) and QuantumClassifier._state_build (D6) must
reproduce today's _vqc_circuit / _vqc_state_circuit exactly, given an
equivalent flat-to-structured weight conversion. This is the safety net
gating D9 (see REFACTOR.md).
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


def _flat_to_circuit_weights(flat: np.ndarray, num_layers: int, n_qubits: int) -> CircuitWeights:
    """Convert today's flat weight vector into the new CircuitWeights layout.

    Mirrors _vqc_circuit's own slicing: weights[-1] is the scale factor,
    weights[-11:-1] are the 10 Hamiltonian coefficients, and the leading
    3*n_qubits*num_layers entries are the per-layer, per-wire (z, y, x)
    rotation triplets, which reshape directly into (L, N, 3).
    """
    rot = flat[: 3 * n_qubits * num_layers].reshape(num_layers, n_qubits, 3)
    aux = {'scale_factor': flat[-1], 'hamiltonian_coeffs': flat[-11:-1]}
    return CircuitWeights(rot=rot, aux=aux)


class TestVQCCircuitParity(unittest.TestCase):
    """VQCCircuit must match QuantumClassifier's private VQC circuit methods."""

    def setUp(self) -> None:
        onp.random.seed(0)
        self.num_layers = 2
        # _vqc_circuit's Hamiltonian always uses 10 coefficients (weights[-11:-1]),
        # so n_qubits must be 10 here or qml.Hamiltonian itself raises -- see
        # AUDIT.md/REFACTOR.md on the fixed-10-vs-N-qubits Hamiltonian coefficients.
        self.n_qubits = 10
        self.wires = list(range(self.n_qubits))
        extra_weights = 11  # 10 Hamiltonian coefficients + 1 scale factor
        n_rotation_weights = 3 * self.n_qubits * self.num_layers
        self.flat_weights = np.array(
            onp.random.uniform(0, np.pi, size=(n_rotation_weights + extra_weights,))
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
        self.qc._impl = VQCCircuit(num_layers=self.num_layers)
        self.circuit_weights = _flat_to_circuit_weights(
            self.flat_weights, self.num_layers, self.n_qubits
        )

    def test_expval_matches(self) -> None:
        """D8: VQCCircuit.build() must match _vqc_circuit's expval exactly."""
        old = qml.QNode(self.qc._vqc_circuit, self.qc.device, interface=self.qc.backend)
        new = qml.QNode(self.qc._impl.build, self.qc.device, interface=self.qc.backend)
        old_out = old(self.flat_weights, self.inputs)
        new_out = new(self.circuit_weights, self.inputs, self.wires)
        self.assertAlmostEqual(float(old_out), float(new_out), places=10)

    def test_state_matches(self) -> None:
        """D6: _state_build must match _vqc_state_circuit's state vector exactly."""
        old = qml.QNode(self.qc._vqc_state_circuit, self.qc.device, interface=self.qc.backend)
        new = qml.QNode(self.qc._state_build, self.qc.device, interface=self.qc.backend)
        old_state = onp.array(old(self.flat_weights, self.inputs))
        new_state = onp.array(new(self.circuit_weights, self.inputs, self.wires))
        self.assertTrue(onp.allclose(old_state, new_state, atol=1e-10))


if __name__ == '__main__':
    unittest.main()
