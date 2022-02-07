import cvxpy as cp
import jax.random
import numpy as np
import numpy.testing as np_test
import pytest
import scipy.linalg as linalg
import scipy.stats as stats
from pytest_mock import MockerFixture

import coupled_rejection_sampling.mvn as coupled_mvns
from coupled_rejection_sampling.mvn import get_optimal_covariance


def is_inverse_less(chol_A, chol_B):
    d, _ = chol_A.shape
    A_inv = linalg.cho_solve((chol_A, True), np.eye(d))
    B_inv = linalg.cho_solve((chol_B, True), np.eye(d))
    assert np.max(np.real(linalg.eigvals(A_inv - B_inv))) > -1e-7


def cvxpy_get_optimal_cov(chol_P, chol_Sig):
    d = chol_P.shape[0]
    eye = np.eye(d)
    P_inv = linalg.cho_solve((chol_P, True), eye)
    Sig_inv = linalg.cho_solve((chol_Sig, True), eye)

    X = cp.Variable((d, d), PSD=True)

    constraints = [X << P_inv, X << Sig_inv]
    objective = cp.log_det(X)
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(warm_start=True, qcp=True, solver=cp.SCS)
    Q_inv = X.value
    chol_Q_inv = linalg.cholesky(Q_inv, lower=True)
    Q = linalg.cho_solve((chol_Q_inv, True), eye)
    chol_Q = np.linalg.cholesky(Q)
    return chol_Q


def test_ordering():
    chol_P = np.array([
        [1.0, 0.0, 0.0],
        [0.2, 0.7, 0.0],
        [0.0, 1.2, 0.5]
    ])

    chol_Sig = np.array([
        [0.2, 0.0, 0.0],
        [0.5, 0.9, 0.0],
        [0.1, 1.7, 1.5]
    ])

    chol_Q = get_optimal_covariance(chol_P, chol_Sig)
    cvxpy_chol_Q = cvxpy_get_optimal_cov(chol_P, chol_Sig)
    np.testing.assert_allclose(chol_Q, cvxpy_chol_Q, rtol=1e-4, atol=1e-5)

    is_inverse_less(chol_P, chol_Q)
    is_inverse_less(chol_Sig, chol_Q)

    assert np.linalg.slogdet(chol_P)[1] < np.linalg.slogdet(chol_Q)[1]
    assert np.linalg.slogdet(chol_Sig)[1] < np.linalg.slogdet(chol_Q)[1]


@pytest.mark.parametrize("d", [1, 2, 3])
def test_mvns_same_cov(d):
    np.random.seed(42)
    key = jax.random.PRNGKey(1234)
    N = 100_000

    m = np.random.randn(d)
    mu = np.random.randn(d)

    chol_Q = 1. + np.random.rand(d, d)
    chol_Q[np.triu_indices(d, 1)] = 0.
    expected_acceptance_ratio = 2 * stats.norm.cdf(
        -0.5 * linalg.norm(linalg.solve_triangular(chol_Q, m - mu, lower=True)))

    # Reflection maximal test
    xs, ys, flags = coupled_mvns.reflection_maximal(key, N, m, mu, chol_Q)

    np_test.assert_allclose(xs.mean(0), m, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(ys.mean(0), mu, atol=1e-2, rtol=1e-2)
    assert np.mean(flags) == pytest.approx(expected_acceptance_ratio, abs=1e-2, rel=1e-2)
    np_test.assert_allclose(xs[flags], ys[flags], atol=1e-2)
    np_test.assert_allclose(np.cov(xs, rowvar=False), chol_Q @ chol_Q.T, atol=1e-2, rtol=1e-2)

    # coupling test
    keys = jax.random.split(key, N)
    xs, ys, flags, n_trials = jax.vmap(lambda k: coupled_mvns.coupled_mvns(k, m, chol_Q, mu, chol_Q, chol_Q=chol_Q))(
        keys)

    np_test.assert_allclose(xs.mean(0), m, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(ys.mean(0), mu, atol=1e-2, rtol=1e-2)
    assert np.mean(flags) == pytest.approx(expected_acceptance_ratio, abs=1e-2, rel=1e-2)
    assert np.max(n_trials) == 1
    np_test.assert_allclose(xs[flags], ys[flags], atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(np.cov(xs, rowvar=False), chol_Q @ chol_Q.T, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("M", [1, 10])
def test_mvns_different_cov(d, M, mocker: MockerFixture):
    spy = mocker.spy(coupled_mvns, "get_optimal_covariance")
    np.random.seed(42)
    key = jax.random.PRNGKey(1234)
    N = 100_000

    m = np.random.randn(d)
    mu = np.random.randn(d)

    chol_P = 1. + 1e-1 * np.random.randn(d, d)
    chol_Sigma = chol_P + 1e-1 * np.random.randn(d, d)

    chol_P[np.triu_indices(d, 1)] = 0.
    chol_Sigma[np.triu_indices(d, 1)] = 0.

    # coupling test
    keys = jax.random.split(key, N)

    xs, ys, flags, n_trials = jax.vmap(lambda k: coupled_mvns.coupled_mvns(k, m, chol_P, mu, chol_Sigma, M))(keys)

    spy.assert_called_once()
    np_test.assert_allclose(xs.mean(0), m, atol=1e-2)
    np_test.assert_allclose(ys.mean(0), mu, atol=1e-2)
    np_test.assert_allclose(xs[flags], ys[flags], atol=1e-2)
    np_test.assert_allclose(np.cov(xs, rowvar=False), chol_P @ chol_P.T, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(np.cov(ys, rowvar=False), chol_Sigma @ chol_Sigma.T, atol=1e-2, rtol=1e-2)


def test_lower_bound():
    # Test against reflection coupling
    m = np.array([1., 2.])
    mu = np.array([1.5, 1.1])

    P = np.array([[2., 1.],
                  [1., 2.]])

    chol_P = linalg.cholesky(P, lower=True)
    K = coupled_mvns.lower_bound(m, chol_P, mu, chol_P, chol_P)
    expected_acceptance_ratio = 2 * stats.norm.cdf(
        -0.5 * linalg.norm(linalg.solve_triangular(chol_P, m - mu, lower=True)))

    assert K == pytest.approx(expected_acceptance_ratio, rel=1e-3, abs=1e-3)


def test_lower_bound_Devroye_et_al():
    # A trivial test
    m = np.array([1.])
    mu = np.array([1.])
    Sig = np.array([[1.]])
    P = np.array([[1.]])

    chol_P = linalg.cholesky(P, lower=True)
    chol_Sig = linalg.cholesky(Sig, lower=True)
    K = coupled_mvns.lower_bound_Devroye_et_al(m, chol_P, mu, chol_Sig)

    assert K == pytest.approx(1., rel=1e-5, abs=1e-5)

    # Test against reflection coupling
    m = np.array([1., 2.])
    mu = np.array([1.5, 1.1])

    P = np.array([[2., 1.],
                  [1., 2.]])

    chol_P = linalg.cholesky(P, lower=True)
    K = coupled_mvns.lower_bound_Devroye_et_al(m, chol_P, mu, chol_P)
    assert K == pytest.approx(0., rel=1e-3, abs=1e-3)
