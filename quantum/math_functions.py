"""
Differentiable loss functions.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import pennylane.numpy as np


def mean_squared_error(predictions,targets):
    """Return squared prediction errors averaged over all array elements."""
    return np.mean((predictions-targets)**2)

def binary_cross_entropy_with_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    """Return stable mean binary cross-entropy from unnormalized logits.

    Labels must lie in [0, 1] and have the same element count as logits;
    they are reshaped to match the logits before averaging over all elements.
    """
    labels = np.reshape(labels, np.shape(logits))
    return np.mean(np.logaddexp(0.0, logits) - labels * logits)
