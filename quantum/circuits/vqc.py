"""
VQC circuit: the variational classifier ansatz migrated from
QuantumClassifier._vqc_circuit into the Circuit protocol.
Author: Aritra Bal (ETP)
2026-09-05
"""
from typing import List, Union

import pennylane as qml
import pennylane.numpy as np

from helpers.utils import getIndex
from quantum.circuits.base import CircuitBase, CircuitWeights


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


class VQCCircuit(CircuitBase):
    """
    Angle-encoded variational classifier: per layer, encode (eta, phi, pt) as
    rotations, entangle with a CNOT ring, then apply a trainable RZ/RY/RX per
    wire. Measured as the expval of a weighted Hamiltonian over all wires.
    """

    operations_per_qubit = 3  # RZ, RY, RX per wire per layer
    aux_defaults = {'scale_factor': 1.0, 'bias': 0.1}
    # 'hamiltonian_coeffs' has no static default here -- its length depends on
    # qubit count, resolved in measure() below.

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'),
            'pt':  getIndex('particle', 'pt'),
        }

    def encode(self, weights: CircuitWeights, inputs: np.ndarray, layer: int, wires: List[int]) -> None:
        n_wires = len(wires)
        sf = 2 * np.pi * sigmoid(weights.aux['scale_factor']) + 1
        for w in wires:
            zenith  = np.squeeze(inputs[:, w + layer * n_wires, self.index['eta']])
            azimuth = np.squeeze(inputs[:, w + layer * n_wires, self.index['phi']])
            radius  = np.squeeze(inputs[:, w + layer * n_wires, self.index['pt']])
            if inputs.shape[0] == 1:
                zenith, azimuth, radius = zenith.item(), azimuth.item(), radius.item()
            qml.RY(sf * radius * zenith, wires=w)
            qml.RX(sf * radius * azimuth, wires=w)

    def entangle(self, wires: List[int]) -> None:
        n_wires = len(wires)
        for w in wires:
            qml.CNOT(wires=[w, (w + 1) % n_wires])

    def rotate(self, weights: CircuitWeights, layer: int, wires: List[int]) -> None:
        rot_z, rot_y, rot_x = (weights.rot[layer, :, i] for i in range(3))
        for rz, ry, rx, w in zip(rot_z, rot_y, rot_x, wires):
            qml.Rot(0., ry, rz, wires=w)
            qml.RX(rx, wires=w)

    def measure(self, weights: CircuitWeights, wires: List[int]) -> qml.measurements.ExpectationMP:
        obs = [qml.PauliZ(i) for i in wires]
        coeffs = weights.aux.get('hamiltonian_coeffs', [0.1] * len(wires))
        if len(coeffs) != len(wires):
            raise ValueError(
                f"hamiltonian_coeffs has {len(coeffs)} entries, circuit has {len(wires)} wires"
            )
        return qml.expval(qml.Hamiltonian(coeffs, obs))
