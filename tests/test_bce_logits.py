"""
Check BCE values and differentiation through circuit scores and bias.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import unittest
from types import SimpleNamespace

import numpy as onp
import pennylane as qml
import pennylane.numpy as np
import torch

from quantum.losses import VQC_cost
from quantum.math_functions import binary_cross_entropy_with_logits


class TestBCELogits(unittest.TestCase):
    """Validate the native Autograd loss against an independent implementation."""

    def test_torch_value_and_gradient_parity(self) -> None:
        for values in ([0.0], [-1000.0, -2.0, 0.0, 2.0, 1000.0]):
            for target in (0.0, 1.0):
                with self.subTest(values=values, target=target):
                    logits = np.array(values)
                    labels = np.full((len(values), 1), target, requires_grad=False)
                    reference = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                    expected = torch.nn.functional.binary_cross_entropy_with_logits(
                        reference, torch.full_like(reference, target)
                    )
                    expected.backward()
                    actual = binary_cross_entropy_with_logits(labels, logits)
                    gradient = qml.grad(lambda z: binary_cross_entropy_with_logits(labels, z))(logits)
                    onp.testing.assert_allclose(actual, expected.item(), atol=1e-12)
                    onp.testing.assert_allclose(gradient, reference.grad.numpy(), atol=1e-12)

    def test_circuit_and_bias_gradients(self) -> None:
        @qml.qnode(qml.device('default.qubit', wires=1), interface='autograd')
        def circuit(weights, inputs):
            qml.RY(weights.rot + inputs, wires=0)
            return qml.expval(qml.PauliZ(0))

        for size in (3, 1):
            inputs = np.array([0.1, 0.4, 0.7][:size], requires_grad=False)
            labels = np.array([0.0, 1.0, 0.0][:size], requires_grad=False)
            params = np.array([0.3, 2.0])

            def cost(parameters, return_scores=False):
                weights = SimpleNamespace(rot=parameters[0], aux={'bias': parameters[1]})
                return VQC_cost(weights, inputs, circuit, labels, return_scores, 'BCE')

            value, scores = cost(params, return_scores=True)
            self.assertTrue(onp.isfinite(value))
            self.assertEqual(scores.shape, (size,))
            onp.testing.assert_allclose(scores, onp.cos(params[0] + inputs) + params[1])
            steps = onp.eye(2) * 1e-6
            expected = [(cost(params + step) - cost(params - step)) / 2e-6 for step in steps]
            onp.testing.assert_allclose(qml.grad(cost)(params), expected, rtol=1e-6, atol=1e-8)
