"""pcasvd: principal component analysis and SVD from scratch.

Jacobi eigen-decomposition, one-sided Jacobi SVD, PCA by both routes,
and image compression by low-rank approximation (Eckart-Young).
"""

from . import compression, data, linalg, pca

__all__ = ["compression", "data", "linalg", "pca"]
__version__ = "1.0.0"
