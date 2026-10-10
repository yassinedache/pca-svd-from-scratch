"""Principal component analysis, computed two ways.

Let X (n x p) be the centred data matrix (optionally standardised).

1. Covariance route:  C = X^T X / (n - 1) = V diag(lambda) V^T (Jacobi diagonalisation).
   The principal axes are the eigenvectors v_j, the variances along them are lambda_j.
2. SVD route:         X = U diag(sigma) V^T (one-sided Jacobi SVD).
   Then C = V diag(sigma^2 / (n - 1)) V^T: same axes V, and lambda_j = sigma_j^2 / (n - 1).
   The scores (coordinates of the points on the axes) are X V = U diag(sigma).

The SVD route never forms X^T X, so it does not square the condition number.
"""

from dataclasses import dataclass

import numpy as np

from .linalg import jacobi_eigen, jacobi_svd


@dataclass
class PCAResult:
    components: np.ndarray      # p x p, column j = j-th principal axis
    variances: np.ndarray       # variance along each axis (decreasing)
    scores: np.ndarray          # n x p, coordinates of the observations
    mean: np.ndarray
    scale: np.ndarray

    @property
    def explained_ratio(self) -> np.ndarray:
        return self.variances / self.variances.sum()

    def transform(self, X: np.ndarray, k: int) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.components[:, :k]


def _prepare(X: np.ndarray, standardize: bool):
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1) if standardize else np.ones(X.shape[1])
    return (X - mean) / scale, mean, scale


def _fix_signs(V: np.ndarray) -> np.ndarray:
    """Eigenvectors are defined up to sign: make the largest entry of each axis positive."""
    signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    signs[signs == 0] = 1
    return V * signs


def pca_covariance(X: np.ndarray, standardize: bool = True) -> PCAResult:
    Z, mean, scale = _prepare(X, standardize)
    C = Z.T @ Z / (len(Z) - 1)
    lam, V = jacobi_eigen(C)
    V = _fix_signs(V)
    return PCAResult(V, np.clip(lam, 0, None), Z @ V, mean, scale)


def pca_svd(X: np.ndarray, standardize: bool = True) -> PCAResult:
    Z, mean, scale = _prepare(X, standardize)
    U, s, Vt = jacobi_svd(Z)
    V = _fix_signs(Vt.T)
    return PCAResult(V, s ** 2 / (len(Z) - 1), Z @ V, mean, scale)
