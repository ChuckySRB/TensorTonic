import numpy as np
from .gan_loss import discriminator_loss, generator_loss
from .generator import generator
from .discriminator import discriminator

def train_gan_step(real_data: np.ndarray, noise_dim: int) -> dict:
    """
    Perform one training step for GAN.
    """
    ###########################
    # 1 - Generate Fake Data
    ###########################
    noise_vec = np.random.normal(0, 1, (real_data.shape[0], noise_dim)).astype(np.float32)
    fake_data = generator(noise_vec, real_data.shape[1])

    #################################################
    # 2 - Classify wich one is fake wich one is real
    #################################################

    real_prediction = discriminator(real_data)
    fake_prediction = discriminator(fake_data)

    ########################
    # 3 - Calculate Loss
    ########################

    Ld = discriminator_loss(real_prediction, fake_prediction)
    Lg = generator_loss(fake_prediction)
    
    return {"d_loss": Ld, "g_loss": Lg}


