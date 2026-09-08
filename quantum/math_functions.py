"""
Differentiable losses and elementwise score transformations.

Author: Aritra Bal (ETP)
Date: 2026-09-08
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

def transform(x,k1=1.0):
    """Map finite scores into (0, 1) with an arctangent scaled by k1."""
    return 0.5*(1+(2./np.pi)*np.arctan(k1*x))



def double_sided_leaky_relu(x):
    """Preserve values in [0, 1] and use slope 0.1 outside that interval."""
    return np.where(x < 0, 0.1 * x, np.where(x > 1, 0.9+0.1*x, x))
