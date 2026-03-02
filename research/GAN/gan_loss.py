import numpy as np

def discriminator_loss(real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
    """
    GAN discriminator loss:
    Ld = -E[log(D(x)) + log(1 - D(G(z)))]
    """
    eps = 1e-7
    real_probs = np.clip(real_probs, eps, 1.0 - eps)
    fake_probs = np.clip(fake_probs, eps, 1.0 - eps)

    ld = -(np.log(real_probs) + np.log(1.0 - fake_probs))
    return float(np.mean(ld))

def generator_loss(fake_probs: np.ndarray) -> float:
    """
    Non-saturating GAN generator loss:
    Lg = -E[log(D(G(z)))]
    """
    eps = 1e-7
    fake_probs = np.clip(fake_probs, eps, 1.0 - eps)

    lg = -np.log(fake_probs)
    return float(np.mean(lg))