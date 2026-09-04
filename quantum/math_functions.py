import pennylane.numpy as np


def mean_squared_error(predictions,targets):
    return np.mean((predictions-targets)**2)

def binary_cross_entropy(labels, probs,bias=None):
    # Ensure that probabilities are bounded between (0, 1)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)  # Avoid log(0) which would cause issues
    # Compute Binary Cross-Entropy
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    # if bias is not None:
    #     loss+=bias
    return loss

def transform(x,k1=1.0):
    return 0.5*(1+(2./np.pi)*np.arctan(k1*x))



def double_sided_leaky_relu(x):
    return np.where(x < 0, 0.1 * x, np.where(x > 1, 0.9+0.1*x, x))