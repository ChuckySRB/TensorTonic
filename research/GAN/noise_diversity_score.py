import numpy as np
from sklearn.metrics import pairwise_distances

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Detect mode collapse in generated samples.
    """

    std = np.std(generated_samples, axis=0)
    div_score = np.mean(std)

    return {
        "diversity_score": div_score,
        "is_collapsed": div_score < threshold   
        }