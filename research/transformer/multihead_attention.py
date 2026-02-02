import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    
    d_model = Q.shape[-1]
    d_k = d_model // num_heads
    batch_size = Q.shape[0]

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