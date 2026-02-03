import numpy as np



def LeakyRelu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.maximum(alpha * x, x)
    
def CreateLinear(input_dim:int, output_dim:int, eps = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    W = np.zeros(shape = (input_dim, output_dim), dtype=np.float32) + eps
    b = np.zeros(shape = (output_dim), dtype=np.float32) + eps
    return W, b

def Linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b

def Tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def Sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))