import numpy as np
import pytest

from pcasvd import compression as comp, data, linalg, pca


@pytest.fixture(scope="module")
def wine():
    return data.load_wine()


def test_wine_shape(wine):
    X, cultivar, names = wine
    assert X.shape == (178, 13) and len(names) == 13 and set(cultivar) == {1, 2, 3}


def test_both_routes_give_the_same_pca(wine):
    X, _, _ = wine
    a, b = pca.pca_covariance(X), pca.pca_svd(X)
    assert np.allclose(a.variances, b.variances, atol=1e-10)
    assert np.allclose(np.abs(np.sum(a.components * b.components, axis=0)), 1, atol=1e-8)
    assert a.explained_ratio.sum() == pytest.approx(1.0)
    # for standardised data the eigenvalues sum to the number of variables
    assert a.variances.sum() == pytest.approx(13.0)


def test_scores_are_uncorrelated_with_the_right_variances(wine):
    X, _, _ = wine
    r = pca.pca_svd(X)
    C = np.cov(r.scores, rowvar=False)
    assert np.allclose(C, np.diag(r.variances), atol=1e-10)


def test_matches_scikit_learn(wine):
    skd = pytest.importorskip("sklearn.decomposition")
    X, _, _ = wine
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    sk = skd.PCA().fit(Z)
    assert np.allclose(sk.explained_variance_, pca.pca_svd(X).variances, atol=1e-10)


def test_eckart_young_error_formula():
    A = np.random.default_rng(3).normal(size=(40, 25))
    U, s, Vt = linalg.jacobi_svd(A)
    for k in (1, 5, 10, 24):
        Ak = linalg.low_rank(U, s, Vt, k)
        assert comp.relative_error(A, Ak) == pytest.approx(comp.eckart_young_error(s, k), abs=1e-12)
        assert np.linalg.norm(A - Ak, 2) == pytest.approx(s[k], rel=1e-10)  # spectral norm = sigma_{k+1}


def test_truncated_svd_beats_other_rank_k_approximations():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(30, 20))
    U, s, Vt = linalg.jacobi_svd(A)
    k = 5
    best = comp.relative_error(A, linalg.low_rank(U, s, Vt, k))
    for _ in range(20):
        Q, _ = np.linalg.qr(rng.normal(size=(30, k)))   # random k-dimensional subspace
        assert comp.relative_error(A, Q @ (Q.T @ A)) >= best - 1e-12  # best rank-k matrix with that column space


def test_storage_and_psnr():
    assert comp.storage_ratio(512, 512, 50) == pytest.approx(50 * 1025 / 512 ** 2)
    A = np.full((4, 4), 0.5)
    assert comp.psnr(A, A) == float("inf")
    assert comp.psnr(A, A + 0.1) == pytest.approx(20.0)
