import math
from functools import partial, lru_cache

import cvxpy as cp
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import jax.random
import jax.scipy.linalg as jlinalg
import jax.scipy.stats as jstats
import numpy as np
import scipy.linalg as linalg
from jax import numpy as jnp
from jax._src.scipy.linalg import cho_solve

from coupled_rejection_sampling.coupled_rejection_sampler import coupled_sampler

_LOG_2PI = math.log(2 * math.pi)


@lru_cache
def _get_cvxpy_pb(d):
    P_inv_param = cp.Parameter((d, d), PSD=True)
    Sig_inv_param = cp.Parameter((d, d), PSD=True)
    z_param = cp.Parameter((d,))

    X = cp.Variable((d, d), PSD=True)

    constraints = [X << P_inv_param, X << Sig_inv_param]
    objective = cp.log_det(X)
    problem = cp.Problem(cp.Maximize(objective), constraints)

    return problem, X, P_inv_param, Sig_inv_param, z_param


def get_optimal_covariance(m, chol_P, mu, chol_Sig, verbose=False):
    """
    Get the optimal covariance according to the objective defined in Section 3 of [1].

    Parameters
    ----------
    m: jnp.ndarray
        Mean of X
    chol_P: jnp.ndarray
        Square root of the covariance of X. Lower triangular.
    mu: jnp.ndarray
        Mean of Y
    chol_Sig: jnp.ndarray
        Square root of the covariance of Y. Lower triangular.
    verbose: bool, optional
        Is the solver verbose. Default is False.
    Returns
    -------
    chol_Q: jnp.ndarray
        Cholesky of the resulting dominating matrix.
    """
    d = m.shape[0]
    if d == 1:
        return np.maximum(chol_P, chol_Sig)
    problem, X, P_inv_param, Sig_inv_param, z_param = _get_cvxpy_pb(d)

    eye = np.eye(d)
    P_inv = linalg.cho_solve((chol_P, True), eye)
    Sig_inv = linalg.cho_solve((chol_Sig, True), eye)

    z_param.value = m - mu
    P_inv_param.value = P_inv
    Sig_inv_param.value = Sig_inv

    problem.solve(warm_start=True, verbose=verbose, qcp=True, solver=cp.MOSEK)

    Q_inv = X.value
    chol_Q_inv = linalg.cholesky(Q_inv, lower=True)
    Q = linalg.cho_solve((chol_Q_inv, True), eye)
    chol_Q = np.linalg.cholesky(Q)
    return chol_Q


def coupled_mvns(key, m, chol_P, mu, chol_Sig, N=1, chol_Q=None):
    """
    Get the optimal covariance according to the objective defined in Section 3 of [1].

    Parameters
    ----------
    key: jnp.ndarray
        JAX random key
    m: array_like
        Mean of X
    chol_P: array_like
        Square root of the covariance of X. Lower triangular.
    mu: array_like
        Mean of Y
    chol_Sig: array_like
        Square root of the covariance of Y. Lower triangular.
    N: int
        Number of samples used in the underlying coupling rejection sampler
    chol_Q: jnp.ndarray, optional
        Square root of the resulting dominating matrix. Default uses get_optimal_covariance.

    Returns
    -------
    X: jnp.ndarray
        The resulting sample for p
    Y: jnp.ndarray
        The resulting sampled for q
    is_coupled: bool
        Do we have X = Y? Note that if the distributions are not continuous this may be False even if X=Y.
    n_trials: int
        The number of trials before acceptance
    """

    if chol_Q is None:
        # This is gonna be slow, but there is no real JAX version...
        chol_Q = hcb.call(lambda args: get_optimal_covariance(*args), (m, chol_P, mu, chol_Sig), result_shape=chol_P)

    log_det_chol_P = tril_log_det(chol_P)
    log_det_chol_Sig = tril_log_det(chol_Sig)
    log_det_chol_Q = tril_log_det(chol_Q)

    log_M_P_Q = jnp.maximum(log_det_chol_Q - log_det_chol_P, 0.)
    log_M_Sigma_Q = jnp.maximum(log_det_chol_Q - log_det_chol_Sig, 0.)

    Gamma_hat = partial(reflection_maximal, m=m, mu=mu, chol_Q=chol_Q)
    log_p = lambda x: mvn_logpdf(x, m, chol_P)
    log_q = lambda x: mvn_logpdf(x, mu, chol_Sig)
    log_p_hat = lambda x: mvn_logpdf(x, m, chol_Q)
    log_q_hat = lambda x: mvn_logpdf(x, mu, chol_Q)
    p = lambda k: mvn_sampler(k, 1, m, chol_P)[0]
    q = lambda k: mvn_sampler(k, 1, mu, chol_Sig)[0]

    return coupled_sampler(key, Gamma_hat, p, q, log_p_hat, log_q_hat, log_p, log_q, log_M_P_Q, log_M_Sigma_Q, N)


@partial(jnp.vectorize, signature="(d),(d),(d,d)->()")
def mvn_logpdf(x, m, chol_P):
    d = m.shape[0]
    log_det_chol_P = tril_log_det(chol_P)
    const = -0.5 * d * _LOG_2PI - log_det_chol_P
    scaled_diff = jlinalg.solve_triangular(chol_P, x - m, lower=True)
    return const - 0.5 * jnp.dot(scaled_diff, scaled_diff)


def mvn_sampler(key, N, m, chol_P):
    d = m.shape[0]
    eps = jax.random.normal(key, (N, d))
    return m[None, :] + eps @ chol_P.T


def tril_log_det(chol):
    return jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))


def reflection_maximal(key, N, m: jnp.ndarray, mu: jnp.ndarray, chol_Q: jnp.ndarray):
    """
    Reflection maximal coupling for Gaussians with the same covariance matrix

    Parameters
    ----------
    key: jnp.ndarray
        The random key for JAX
    N: int
        Number of samples required
    m:

    mu
    chol_Q

    Returns
    -------

    """
    dim = m.shape[0]
    z = jlinalg.solve_triangular(chol_Q, m - mu, lower=True)
    e = z / jnp.linalg.norm(z)

    normal_key, uniform_key = jax.random.split(key, 2)
    norm = jax.random.normal(normal_key, (N, dim))
    log_u = jnp.log(jax.random.uniform(uniform_key, (N,)))

    temp = norm + z[None, :]

    mvn_loglikelihood = lambda x: - 0.5 * jnp.sum(x ** 2, -1)

    do_accept = log_u + mvn_loglikelihood(norm) < mvn_loglikelihood(temp)

    reflected_norm = jnp.where(do_accept[:, None], temp, norm - 2 * jnp.outer(jnp.dot(norm, e), e))

    res_1 = m[None, :] + norm @ chol_Q.T
    res_2 = mu[None, :] + reflected_norm @ chol_Q.T

    return res_1, res_2, do_accept


def lower_bound(m: jnp.ndarray, chol_P: jnp.ndarray, mu: jnp.ndarray, chol_Sig: jnp.ndarray, chol_Q: jnp.ndarray):
    """

    Parameters
    ----------
    m
    chol_P
    mu
    chol_Sig
    chol_Q

    Returns
    -------

    """
    d = m.shape[0]
    eye = jnp.eye(d)
    iP = jlinalg.cho_solve((chol_P, True), eye)
    iSig = jlinalg.cho_solve((chol_Sig, True), eye)
    iQ = jlinalg.cho_solve((chol_Q, True), eye)

    iH = iP + iSig - iQ
    chol_iH = jlinalg.cholesky(iH, lower=True)
    H = jlinalg.cho_solve((chol_iH, True), eye)

    a = H @ (iP @ m + (iSig - iQ) @ mu)
    b = m.T @ iP @ m + mu.T @ (iSig - iQ) @ mu - a.T @ iH @ a
    d = H @ (iSig @ mu + (iP - iQ) @ m)
    g = mu.T @ iSig @ mu + m.T @ (iP - iQ) @ m - d.T @ iH @ d

    # Avoid division by zero in F
    jitter = 1e-7
    m = jax.lax.select(jnp.linalg.norm(m - mu) < jitter, m + jitter, m)

    def F(u: jnp.ndarray, chol_V: jnp.ndarray):
        num = 0.5 * (m.T @ iQ @ m - mu.T @ iQ @ mu - 2 * u.T @ iQ @ (m - mu))
        den = jnp.linalg.norm(chol_V.T @ iQ @ (m - mu))
        return jstats.norm.cdf(num / den)

    chol_H = jlinalg.cholesky(H, lower=True)

    acceptance_part = jnp.exp(tril_log_det(chol_H) - tril_log_det(chol_Q))
    coupling_part = jnp.exp(-b / 2) * F(a, chol_H) + jnp.exp(-g / 2) * (1 - F(d, chol_H))
    return acceptance_part * coupling_part


def lower_bound_Devroye_et_al(m: jnp.ndarray, chol_P: jnp.ndarray, mu: jnp.ndarray, chol_Sig: jnp.ndarray):
    """
    Theorem 1.2 in https://arxiv.org/abs/1810.08693

    Parameters
    ----------
    m
    chol_P
    mu
    chol_Sig

    Returns
    -------

    """
    d = m.shape[0]
    v = m - mu
    jitter = 1e-6
    v = jax.lax.select(jnp.linalg.norm(v) < jitter, jitter * jnp.ones_like(v), v)

    P = chol_P @ chol_P.T
    Sig = chol_Sig @ chol_Sig.T

    if d > 1:
        # complete an orthonormal basis given by vectors orthogonal to v
        U, *_ = jlinalg.svd(v[:, None])
        Pi = U[:, 1:]

        # compute eigenvals of auxiliary matrix
        aux = jlinalg.solve(Pi.T @ P @ Pi, Pi.T @ Sig @ Pi, sym_pos=True) - jnp.eye(d - 1)
        chol_aux = jlinalg.cholesky(aux)
        eig_vals = jnp.nan_to_num(jnp.diag(chol_aux))
    else:
        eig_vals = jnp.array([0.])

    # Compute the lowercase TV
    tv_1 = jnp.abs(jnp.dot(v, (Sig - P) @ v)) / jnp.dot(v, (P @ v))
    tv_2 = jnp.dot(v, v) / jnp.dot(v, (P @ v)) ** 0.5
    tv_3 = jnp.sum(eig_vals ** 2) ** 0.5
    tv = jnp.maximum(jnp.maximum(tv_1, tv_2), tv_3)

    return jnp.maximum(1 - 4.5 * jnp.minimum(tv, 1), 0)


def compute_asymptotic_bound(chol_P, chol_Q, chol_Sigma):
    """
    Computes the asymptotic lower bound as a function of N for Gaussians

    Parameters
    ----------
    chol_P
    chol_Q
    chol_Sigma

    Returns
    -------

    """
    d = chol_P.shape[0]
    eye = jnp.eye(d)

    def stddev_term(chol_A, chol_B):
        # Note that the 2 pi constant cancels out.
        B_part = tril_log_det(chol_Q)
        A_part = -2 * tril_log_det(chol_P)

        A_inv = cho_solve((chol_A, True), eye)
        B_inv = cho_solve((chol_B, True), eye)

        temp = jnp.linalg.cholesky(2 * A_inv - B_inv)
        AB_part = tril_log_det(temp)
        return jnp.sqrt(jnp.exp(A_part + B_part + AB_part) - 1)

    var_P_Q = stddev_term(chol_P, chol_Q)
    var_Sig_Q = stddev_term(chol_Sigma, chol_Q)
    upper_std = jnp.maximum(var_P_Q, var_Sig_Q)
    lower_std = jnp.minimum(var_P_Q, var_Sig_Q)
    M_P = jnp.exp(tril_log_det(chol_Q) - tril_log_det(chol_P))
    M_Sig = jnp.exp(tril_log_det(chol_Q) - tril_log_det(chol_Sigma))

    @partial(jnp.vectorize, signature="()->(m)")
    def bound_func(N):
        A = N / (N - 1 + M_P)
        B = N / (N - 1 + M_Sig)
        C = jnp.sqrt(2 * N * jnp.log(jnp.log(N))) / N

        return jnp.array([A * B / (1 + lower_std * C), 1. / (1. - upper_std * C)])

    return bound_func


def coupling_probability_same_cov(m, mu, chol_Q):
    """
    Computes the coupling probability of a maximal coupling between N(m, Q) and N(mu, Q)

    Parameters
    ----------
    m
    mu
    chol_Q

    Returns
    -------

    """
    return 2 * jstats.norm.cdf(-0.5 * jnp.linalg.norm(jlinalg.solve_triangular(chol_Q, m - mu, lower=True)))
