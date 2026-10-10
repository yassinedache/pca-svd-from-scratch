"""Eigen-decomposition and SVD written from scratch, with plane rotations (Jacobi methods).

jacobi_eigen  symmetric S = V diag(lambda) V^T. Each rotation zeroes one off-diagonal
              entry; the off-diagonal mass sum_{i != j} s_ij^2 decreases at every step,
              so S converges to a diagonal matrix.
jacobi_svd    one-sided Jacobi (Hestenes): rotate pairs of columns of A until all columns
              are orthogonal. Then A V = U Sigma: the column norms are the singular values.
              Pairs are processed in a round-robin order so that n/2 disjoint rotations are
              applied at once with NumPy, which makes it fast enough for a 512 x 512 image.
"""

import numpy as np


def jacobi_eigen(S: np.ndarray, tol: float = 1e-14, max_sweeps: int = 100):
    """Eigenvalues (in decreasing order) and orthonormal eigenvectors of a symmetric matrix."""
    A = np.array(S, dtype=float, copy=True)
    n = A.shape[0]
    V = np.eye(n)
    scale = np.linalg.norm(A)
    for _ in range(max_sweeps):
        off = np.sqrt(max(np.sum(A ** 2) - np.sum(np.diag(A) ** 2), 0.0))
        if off <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < 1e-300:
                    continue
                # rotation angle that zeroes A[p, q]: tan(2 theta) = 2 a_pq / (a_qq - a_pp)
                tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
                if tau == 0:
                    t = 1.0
                elif abs(tau) > 1e8:  # avoids overflow of tau^2; t ~ 1 / (2 tau)
                    t = 1 / (2 * tau)
                else:
                    t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau ** 2))
                c = 1 / np.sqrt(1 + t ** 2)
                s = t * c
                # A <- J^T A J, V <- V J
                Ap, Aq = A[:, p].copy(), A[:, q].copy()
                A[:, p], A[:, q] = c * Ap - s * Aq, s * Ap + c * Aq
                Ap, Aq = A[p, :].copy(), A[q, :].copy()
                A[p, :], A[q, :] = c * Ap - s * Aq, s * Ap + c * Aq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p], V[:, q] = c * Vp - s * Vq, s * Vp + c * Vq
    eigvals = np.diag(A).copy()
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], V[:, order]


def _round_robin(n: int):
    """Yield n - 1 (or n) rounds of disjoint pairs covering every pair (p, q) once (circle method)."""
    players = list(range(n)) + ([-1] if n % 2 else [])
    m = len(players)
    for _ in range(m - 1):
        pairs = [(players[i], players[m - 1 - i]) for i in range(m // 2)]
        yield [(min(a, b), max(a, b)) for a, b in pairs if a >= 0 and b >= 0]
        players = [players[0]] + [players[-1]] + players[1:-1]


def jacobi_svd(A: np.ndarray, tol: float = 1e-13, max_sweeps: int = 60):
    """Thin SVD A = U diag(s) V^T of an m x n matrix with m >= n (s in decreasing order)."""
    U = np.array(A, dtype=float, copy=True)
    m, n = U.shape
    if m < n:
        V_, s, U_ = jacobi_svd(np.asarray(A, dtype=float).T, tol, max_sweeps)
        return U_.T, s, V_.T
    W = np.ascontiguousarray(U.T)  # row i of W is column i of U (contiguous in memory)
    Z = np.eye(n)                   # row i of Z is column i of V
    rounds = [np.array(r, dtype=int) for r in _round_robin(n)]
    for _ in range(max_sweeps):
        worst = 0.0
        for r in rounds:
            if r.size == 0:
                continue
            P, Q = r[:, 0], r[:, 1]
            up, uq = W[P], W[Q]
            alpha = np.einsum("ij,ij->i", up, up)
            beta = np.einsum("ij,ij->i", uq, uq)
            gamma = np.einsum("ij,ij->i", up, uq)
            denom = np.sqrt(alpha * beta)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(denom > 0, np.abs(gamma) / denom, 0.0)
            worst = max(worst, float(ratio.max(initial=0.0)))
            act = ratio > tol
            if not act.any():
                continue
            P, Q, alpha, beta, gamma = P[act], Q[act], alpha[act], beta[act], gamma[act]
            zeta = (beta - alpha) / (2 * gamma)
            t = np.sign(zeta) / (np.abs(zeta) + np.sqrt(1 + zeta ** 2))
            t[zeta == 0] = 1.0
            c = (1 / np.sqrt(1 + t ** 2))[:, None]
            s = c * t[:, None]
            up, uq = W[P], W[Q]
            W[P], W[Q] = c * up - s * uq, s * up + c * uq
            vp, vq = Z[P], Z[Q]
            Z[P], Z[Q] = c * vp - s * vq, s * vp + c * vq
        if worst <= tol:
            break
    U, V = W.T, Z.T
    sing = np.linalg.norm(U, axis=0)
    order = np.argsort(sing)[::-1]
    sing, U, V = sing[order], U[:, order], V[:, order]
    nz = sing > sing[0] * 1e-15 if sing.size and sing[0] > 0 else np.zeros(n, bool)
    U[:, nz] /= sing[nz]
    U[:, ~nz] = 0.0
    return U, sing, V.T


def low_rank(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation A_k = sum_{i <= k} s_i u_i v_i^T (Eckart-Young)."""
    return (U[:, :k] * s[:k]) @ Vt[:k, :]
