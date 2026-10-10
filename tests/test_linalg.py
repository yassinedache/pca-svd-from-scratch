import numpy as np
import pytest

from pcasvd import linalg


@pytest.mark.parametrize("n", [2, 5, 13])
def test_jacobi_eigen(n):
    B = np.random.default_rng(n).normal(size=(n, n))
    S = B @ B.T
    lam, V = linalg.jacobi_eigen(S)
    assert np.all(np.diff(lam) <= 1e-12)                       # decreasing order
    assert np.allclose(lam, np.sort(np.linalg.eigvalsh(S))[::-1], atol=1e-10)
    assert np.allclose(V.T @ V, np.eye(n), atol=1e-12)         # orthonormal
    assert np.allclose(S @ V, V * lam, atol=1e-10)             # S v = lambda v


@pytest.mark.parametrize("shape", [(20, 5), (178, 13), (7, 30), (64, 64)])
def test_jacobi_svd(shape):
    A = np.random.default_rng(sum(shape)).normal(size=shape)
    U, s, Vt = linalg.jacobi_svd(A)
    k = min(shape)
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False), atol=1e-11)
    assert np.allclose((U * s) @ Vt, A, atol=1e-11)
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-11)
    assert np.allclose(Vt @ Vt.T, np.eye(k), atol=1e-11)


def test_svd_of_rank_deficient_matrix():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 3)) @ rng.normal(size=(3, 8))   # rank 3
    U, s, Vt = linalg.jacobi_svd(A)
    assert np.all(s[3:] < 1e-12 * s[0])
    assert np.allclose((U * s) @ Vt, A, atol=1e-11)


def test_round_robin_covers_every_pair_once():
    for n in (6, 7):
        pairs = [p for r in linalg._round_robin(n) for p in r]
        assert sorted(pairs) == [(i, j) for i in range(n) for j in range(i + 1, n)]
