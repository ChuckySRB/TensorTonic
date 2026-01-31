import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    sigmoid = 1 / (1 + np.exp(-np.asarray(x, dtype = float)))
    return sigmoid