import math

import jax.numpy as jnp
import jax.random
import numpy as np

from coupled_rejection_sampling.coupled_rejection_sampler import coupled_sampler
from coupled_rejection_sampling.rejection_sampler import rejection_sampler, accept_proposal_and_acceptance_ratio

_LOG_HALF = math.log(0.5)


def coupled_parallel_resampler(key: jnp.ndarray,
                               log_w_X: jnp.ndarray,
                               log_w_Y: jnp.ndarray,
                               log_M_X: float,
                               log_M_Y: float,
                               N: int = 1):
    """
    A coupled version of the parallel resampler described in Algorithm 3 of [3].
    The algorithm is described in Section xxx of [1].

    Parameters
    ----------
    key: jnp.ndarray
       JAX random key
    log_w_X: jnp.ndarray
        Unnormalised log weights for X to resample
    log_w_Y: jnp.ndarray
       Unnormalised log weights for Y to resample
    log_M_X, log_M_Y: float
        Upper bound for the weights
    N: int, optional
        The number of particles to be used in the ensemble. Default is 1, which reduces to simple coupled
        rejection sampling.

    Returns
    -------
    A_X: jnp.ndarray
        The resulting ancestors for X
    A_Y: jnp.ndarray
        The resulting ancestors for Y
    n_trials: jnp.ndarray
        The number of trials before acceptance for each index
    is_coupled: jnp.ndarray
        Coupling flag for each index

    References
    ----------
    .. [1]
    """

    n_particles = log_w_X.shape[0]
    keys = jax.random.split(key, n_particles)
    arange = np.arange(n_particles)
    spec_coupled_resampler_one = lambda k, m: _coupled_resampler_one(k, m, log_w_X, log_w_Y, log_M_X, log_M_Y, N)
    return jax.vmap(spec_coupled_resampler_one)(keys, arange)


def _coupled_resampler_one(key: jnp.ndarray,
                           m: int,
                           log_w_X: jnp.ndarray,
                           log_w_Y: jnp.ndarray,
                           log_M_X: float,
                           log_M_Y: float,
                           N: int = 1):
    n_particles = log_w_X.shape[0]
    X = Y = jnp.array([m])

    common_prop = lambda k, M: jax.random.randint(k, (M,), 0, n_particles)

    def Gamma_hat(k, M):
        idx = common_prop(k, M)
        return idx, idx, jnp.ones((M,), bool)

    log_prop = lambda idx: jnp.ones_like(idx)
    log_p = lambda idx: jnp.take(log_w_X, idx, 0)
    log_q = lambda idx: jnp.take(log_w_Y, idx, 0)

    def p_rejection_sampler(op_k):
        indices, _ = rejection_sampler(op_k, common_prop, log_prop, log_p, log_M_X, N)
        return indices

    def q_rejection_sampler(op_k):
        indices, _ = rejection_sampler(op_k, common_prop, log_prop, log_q, log_M_Y, N)
        return indices

    init_key, rest_key = jax.random.split(key)

    # SAME KEY !!!! DO NOT CHANGE OR I WON'T BE HAPPY. YEAH, I'M TALKING TO YOU, FUTURE ME!
    accept_X, X = accept_proposal_and_acceptance_ratio(init_key, X, log_p, log_prop, log_M_X, N=1)
    accept_Y, Y = accept_proposal_and_acceptance_ratio(init_key, Y, log_q, log_prop, log_M_Y, N=1)

    stop_here = accept_X & accept_Y

    def if_stop_here(_op_key):
        return X, Y, True, 1

    def otherwise(op_key):
        return coupled_sampler(op_key, Gamma_hat,
                               p_rejection_sampler, q_rejection_sampler,
                               log_prop, log_prop,
                               log_p, log_q,
                               log_M_X, log_M_Y, N)

    # This need to take into account the time taken to run the individual rejection sampler too...
    X_final, Y_final, _, n_trials = jax.lax.cond(stop_here, if_stop_here, otherwise, rest_key)
    is_coupled = X_final == Y_final
    return X_final, Y_final, is_coupled, n_trials
