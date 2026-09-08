import pennylane.numpy as np


def mean_squared_error(predictions,targets):
    return np.mean((predictions-targets)**2)

def binary_cross_entropy_with_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    """Compute stable mean BCE directly from differentiable circuit logits."""
    labels = np.reshape(labels, np.shape(logits))
    return np.mean(np.logaddexp(0.0, logits) - labels * logits)

def transform(x,k1=1.0):
    return 0.5*(1+(2./np.pi)*np.arctan(k1*x))



def double_sided_leaky_relu(x):
    return np.where(x < 0, 0.1 * x, np.where(x > 1, 0.9+0.1*x, x))
