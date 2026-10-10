"""Image compression by low-rank approximation.

Eckart-Young-Mirsky theorem: among all matrices B of rank <= k,
    ||A - B||_F is minimal for B = A_k = sum_{i<=k} sigma_i u_i v_i^T, and
    ||A - A_k||_F = sqrt(sigma_{k+1}^2 + ... + sigma_r^2),   ||A - A_k||_2 = sigma_{k+1}.

Storing A_k needs k (m + n + 1) numbers instead of m n.
"""

import numpy as np


def relative_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


def eckart_young_error(s: np.ndarray, k: int) -> float:
    """Relative Frobenius error of the best rank-k approximation, from the singular values only."""
    return float(np.sqrt(np.sum(s[k:] ** 2) / np.sum(s ** 2)))


def storage_ratio(m: int, n: int, k: int) -> float:
    """Numbers stored for A_k divided by numbers in A."""
    return k * (m + n + 1) / (m * n)


def psnr(A: np.ndarray, B: np.ndarray, peak: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB (images with values in [0, peak])."""
    mse = float(np.mean((A - np.clip(B, 0, peak)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)
