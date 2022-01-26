import math
from typing import Callable

import jax.numpy as jnp
import jax.random
from jax.scipy.special import logsumexp

from coupled_rejection_sampling.utils import logsubexp

_LOG_HALF = math.log(0.5)


def rejection_sampler(key: jnp.ndarray,
                      p_hat: Callable,
                      log_p_hat: Callable,
                      log_p: Callable,
                      log_M_p: float,
                      N: int = 1):
    """
    This is the general code for Algorithm 1 of the ensemble rejection method paper [2]. In the case when `N` it
    reduces to a standard rejection sampler.

    Parameters
    ----------
    key: jnp.ndarray
       JAX random key
    p_hat: callable
        Sample from the proposal
    log_p_hat, log_p: callable
        The log densities of the dominating and target densities. They take arrays (N, d) and return an array (N,).
    log_M_p: float
        Logarithm of the dominating constant for log_p and log_p_hat: log_p < log_M_p + log_p_hat
    N: int, optional
        The number of particles to be used in the ensemble. Default is 1, which reduces to simple coupled
        rejection sampling.

    Returns
    -------
    X: jnp.ndarray
        The resulting sample for p
    n_trials: int
        The number of trials before acceptance

    References
    ----------
    .. [1]
    """

    def cond(carry):
        accept_X, *_ = carry
        return ~accept_X

    def body(carry):
        *_, i, curr_key = carry
        next_key, sample_key, accept_key = jax.random.split(curr_key, 3)
        Xs_hat = p_hat(sample_key, N)

        accept_X, X_hat = accept_proposal_and_acceptance_ratio(accept_key, Xs_hat, log_p, log_p_hat, log_M_p, N)
        return accept_X, X_hat, i + 1, next_key

    # initialisation to get shape values
    init_key, key = jax.random.split(key)
    X_init = p_hat(init_key, 1)[0]

    output = jax.lax.while_loop(cond,
                                lambda carry: body(carry),
                                (False, X_init, 0, key))

    _, X, n_trials, _ = output

    return X, n_trials


def accept_proposal_and_acceptance_ratio(op_key, Xs_hat, log_p, log_p_hat, log_M_p, N):
    """
    This is the general code for Algorithm 3 of the coupled rejection method paper [1]. In the case when `N` it
    reduces to Algorithm 1.

    Parameters
    ----------
    op_key: jnp.ndarray
       JAX random key
    Xs_hat: jnp.ndarray
        Samples from the proposal
    log_p, log_p_hat: callable
        The log densities of the dominating and target densities. They take arrays (N, d) and return an array (N,).
    log_M_p: float
        Logarithm of the dominating constant for log_p and log_p_hat: log_p < log_M_p + log_p_hat
    N: int, optional
        The number of particles to be used in the ensemble. Default is 1, which reduces to simple coupled
        rejection sampling.

    Returns
    -------
    accept_X: bool
        Was X accepted
    X_hat: jnp.ndarray
        The chosen sample that was or not accepted.

    """
    # unnormalised log weights
    select_key, accept_key = jax.random.split(op_key, 2)
    log_w_X = log_p(Xs_hat) - log_p_hat(Xs_hat)

    if N == 1:
        X_hat = Xs_hat[0]
        X_acceptance_proba = log_w_X[0] - log_M_p
    else:
        log_N = math.log(N)

        # log likelihood of the samples
        log_Z_X_hat = logsumexp(log_w_X) - log_N

        # Normalised log weights
        log_W_X = log_w_X - log_N - log_Z_X_hat

        # index sampling
        I = jax.random.choice(select_key, N, p=jnp.exp(log_W_X))

        # Select the proposal
        X_hat = Xs_hat[I]

        # Compute the upper bounds
        log_Z_X_bar = jnp.logaddexp(log_Z_X_hat, logsubexp(log_M_p, log_w_X[I]) - log_N)

        X_acceptance_proba = log_Z_X_hat - log_Z_X_bar

    log_u = jnp.log(jax.random.uniform(accept_key))
    accept_X = log_u < X_acceptance_proba
    return accept_X, X_hat
