import numpy as np

class GAN:
    def __init__(self, data_dim: int, noise_dim: int):
        self.data_dim = data_dim
        self.noise_dim = noise_dim

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate fake data from noise vectors.
        Output shape: (n_samples, data_dim)
        """
        # z: (batch, noise_dim)
        z = np.random.normal(0, 1, (n_samples, self.noise_dim)).astype(np.float32)

        # Layer 1: noise_dim -> 128
        W1, b1 = CreateLinear(self.noise_dim, 128)
        x = LeakyRelu(Linear(z, W1, b1))

        # Layer 2: 128 -> 256
        W2, b2 = CreateLinear(128, 256)
        x = LeakyRelu(Linear(x, W2, b2))

        # Layer 3: 256 -> data_dim
        W3, b3 = CreateLinear(256, self.data_dim)
        x = Linear(x, W3, b3)

        # Output
        return Tanh(x)

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """
        Classify inputs as real or fake.
        Expected input shape: (batch, data_dim)
        Output shape: (batch, 1)
        """
        W1, b1 = CreateLinear(self.data_dim, 256)
        x = LeakyRelu(Linear(x, W1, b1))

        W2, b2 = CreateLinear(256, 128)
        x = LeakyRelu(Linear(x, W2, b2))

        W3, b3 = CreateLinear(128, 1)
        x = Linear(x, W3, b3)

        return Sigmoid(x)

    def train_step(self, real_data: np.ndarray) -> dict:
        fake_data = self.generate(real_data.shape[0])

        real_prediction = self.discriminate(real_data)
        fake_prediction = self.discriminate(fake_data)

        Ld = self.discriminator_loss(real_prediction, fake_prediction)
        Lg = self.generator_loss(fake_prediction)

        return {"d_loss": Ld, "g_loss": Lg}

    def discriminator_loss(self, real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
        eps = 1e-7
        real_probs = np.clip(real_probs, eps, 1.0 - eps)
        fake_probs = np.clip(fake_probs, eps, 1.0 - eps)
        ld = -(np.log(real_probs) + np.log(1.0 - fake_probs))
        return float(np.mean(ld))

    def generator_loss(self, fake_probs: np.ndarray) -> float:
        eps = 1e-7
        fake_probs = np.clip(fake_probs, eps, 1.0 - eps)
        lg = -np.log(fake_probs)
        return float(np.mean(lg))


def LeakyRelu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.maximum(alpha * x, x)

def CreateLinear(input_dim: int, output_dim: int, eps=1e-6) -> tuple[np.ndarray, np.ndarray]:
    W = np.zeros((input_dim, output_dim), dtype=np.float32) + eps
    b = np.zeros((output_dim,), dtype=np.float32) + eps
    return W, b

def Linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b

def Tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def Sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
