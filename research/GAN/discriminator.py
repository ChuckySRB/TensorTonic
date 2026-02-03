import numpy as np
from research.GAN.neural_networks import CreateLinear, Linear, LeakyRelu, Sigmoid

def discriminator(x: np.ndarray) -> np.ndarray:
    """
    Classify inputs as real or fake.
    """
    
    ###########
    # Layer 1 #
    ###########
    W1, b1 = CreateLinear(x.shape[1], 256)
    x = Linear(x, W1, b1)
    x = LeakyRelu(x)

    ###########
    # Layer 2 #
    ###########
    W2, b2 = CreateLinear(256, 128)
    x = Linear(x, W2, b2)
    x = LeakyRelu(x)

    ###########
    # Layer 3 #
    ###########
    W3, b3 = CreateLinear(128, 1)
    x = Linear(x, W3, b3)

    ################
    # Output Layer #
    ################
    x = Sigmoid(x)

    return x