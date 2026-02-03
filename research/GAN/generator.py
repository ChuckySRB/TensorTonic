import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    # Your implementation here

    rgbt, w = z.shape

    eps = 1e-6

    W = np.zeros(shape = (w, output_dim), dtype=np.float32) + eps
    b = np.zeros(shape = (output_dim), dtype=np.float32) + eps

    x = np.tanh(z @ W + b)

    return x