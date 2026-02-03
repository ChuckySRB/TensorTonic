import numpy as np
from research.GAN.neural_networks import CreateLinear, Linear, LeakyRelu, Tanh


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