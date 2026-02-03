import numpy as np



def LeakyRelu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.2 * x, x)
    
def CreateLinear(input_dim:int, output_dim:int, eps = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    W = np.zeros(shape = (input_dim, output_dim), dtype=np.float32) + eps
    b = np.zeros(shape = (output_dim), dtype=np.float32) + eps
    return W, b

def Linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b

def Tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    ###########
    # Layer 1 #
    ###########
    W1, b1 = CreateLinear(z.shape[1], 128)
    x = Linear(z, W1, b1)
    x = LeakyRelu(x)

    ###########
    # Layer 2 #
    ###########
    W2, b2 = CreateLinear(128, 256)
    x = Linear(x, W2, b2)
    x = LeakyRelu(x)

    ###########
    # Layer 3 #
    ###########
    W3, b3 = CreateLinear(256, output_dim)
    x = Linear(x, W3, b3)
   
    ################
    # Output Layer #
    ################
    x = Tanh(x)

    return x