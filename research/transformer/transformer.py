import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    LN = gamma * (x - mean) / np.sqrt(variance + eps) + beta 
    return LN

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    d_model = Q.shape[-1]
    d_k = d_model // num_heads
    batch_size = Q.shape[0]
    N = Q.shape[1]


    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    Q_h = Q_proj.reshape(batch_size, N, num_heads, d_k).transpose(0, 2, 1, 3)
    K_h = K_proj.reshape(batch_size, N, num_heads, d_k).transpose(0, 2, 1, 3)
    V_h = V_proj.reshape(batch_size, N, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Scores and Attention for each head
    scores = Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    weghts = softmax(scores, axis=-1)
    A_h = weghts @ V_h
    attention = A_h.transpose(0, 2, 1, 3).reshape(batch_size, N, d_model)
    multihead = attention @ W_o

    return multihead

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.maximum(0, x @ W1 + b1)  # ReLU activation
    output = hidden @ W2 + b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Multi-head attention
    mha_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    # Add & Norm
    x1 = layer_norm(x + mha_out, gamma1, beta1)
    # Feed-forward network
    ffn_out = feed_forward(x1, W1, b1, W2, b2)
    # Add & Norm
    out = layer_norm(x1 + ffn_out, gamma2, beta2)

    return out