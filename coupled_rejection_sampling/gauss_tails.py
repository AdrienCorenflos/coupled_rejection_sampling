# Coupled sampling from 2 different tails x > mu and x > eta of N(x | 0,1) Gaussian
# by using a maximal coupling of translated exponential proposals hatp(x) = alpha exp(-alpha (x - mu))
# and hatq(x) = beta exp(-beta (x - eta)) similarly to Robert (1995), but with the coupling.
#
# Robert, C. P. (1995). Simulation of truncated normal variables. Statistics and Computing, Volume 5, pages 121–125.
import chex as chex
import jax.lax
import jax.numpy as jnp
import jax.random
import jax.scipy.stats as jstats
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.host_callback import id_print

from coupled_rejection_sampling.coupled_rejection_sampler import coupled_sampler
from coupled_rejection_sampling.utils import logsubexp, log1mexp


def coupled_gaussian_tails(key: jnp.ndarray,
                           mu: float, eta: float, N: int = 1):
    """
    A coupled version of Robert's truncated normal sampling algorithm.
    We want to sample from 2 different tails x > mu and x > eta of a N(x | 0,1) Gaussian

    Parameters
    ----------
    key: jnp.ndarray
        JAX random key
    mu, eta: float
        The tails of the Gaussians we want to sample from
    N: int, optional
        The number of particles to be used in the ensemble. Default is 1, which reduces to simple coupled
        rejection sampling.

    Returns
    -------
    X: float
        The sample from x > mu
    Y: float
        The sample from x > eta
    n_trials: int
        The number of trials before acceptance
    is_coupled: bool
        Coupling flag
    """
    chex.assert_scalar_positive(eta - mu)
    alpha_mu = get_alpha(mu)
    alpha_eta = get_alpha(eta)

    vmapped_coupled_sampler = jax.vmap(lambda k: coupled_exponentials(k, mu, alpha_mu, eta, alpha_eta))

    def Gamma_hat(k, M):
        keys = jax.random.split(k, M)
        return vmapped_coupled_sampler(keys)

    p = lambda k: _robert_sampler(k, mu, alpha_mu)
    q = lambda k: _robert_sampler(k, eta, alpha_eta)

    log_p = lambda x: -0.5 * (x - alpha_mu) ** 2
    log_q = lambda x: -0.5 * (x - alpha_eta) ** 2

    log_pq_hat = lambda x: jnp.zeros_like(x)

    # ATTN: We actually put log_p = log( p / p_hat ), log_p_hat = 0, M_p = 1 and the same for q
    # to avoid explicit evaluation of p and p_hat and M.
    return coupled_sampler(key, Gamma_hat, p, q, log_pq_hat, log_pq_hat, log_p, log_q, 0., 0., N)


def coupled_exponentials(key, mu, alpha_mu, eta, alpha_eta):
    chex.assert_scalar_positive(eta - mu)

    gamma = get_gamma(mu, eta, alpha_mu, alpha_eta)

    eta_mu = -alpha_mu * (eta - mu)
    gamma_mu = -alpha_mu * (gamma - mu)
    gamma_eta = -alpha_eta * (gamma - eta)

    log_max_coupling_proba = logsubexp(eta_mu, jnp.logaddexp(gamma_mu, gamma_eta))

    subkey1, subkey2 = jax.random.split(key)

    log_u = jnp.log(jax.random.uniform(subkey1, shape=()))
    are_coupled = (log_u <= log_max_coupling_proba)

    def if_coupled(k):
        x = _sampled_from_coupled_exponentials(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        return x, x

    def otherwise(k):
        x = _sample_from_first_marginal(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        y = _sample_from_second_marginal(k, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma)
        return x, y

    return *jax.lax.cond(are_coupled, if_coupled, otherwise, subkey2), are_coupled


def _sampled_from_coupled_exponentials(key, mu, _eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    def C1_inv(u):
        log_u = jnp.log(u)
        return mu - logsubexp(eta_mu, log_u + logsubexp(eta_mu, gamma_mu)) / alpha_mu

    def C2_inv(u):
        return gamma - jnp.log(1.0 - u) / alpha_eta

    log_p1 = logsubexp(eta_mu, gamma_mu)
    log_p2 = gamma_eta
    log_p = log_p1 - jnp.logaddexp(log_p1, log_p2)

    u1, u2 = jax.random.uniform(key, shape=(2,))

    res = jax.lax.cond(jnp.log(u1) < log_p, C1_inv, C2_inv, u2)
    return res


def _sample_from_first_marginal(key, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    key1, key2 = jax.random.split(key, 2)
    log_u1 = jnp.log(jax.random.uniform(key1))

    log_p1 = logsubexp(gamma_mu, gamma_eta)
    log_p2 = log1mexp(eta_mu)
    log_p = log_p1 - jnp.logaddexp(log_p1, log_p2)

    log_tZ = logsubexp(gamma_mu, gamma_eta)

    def _sample_from_tail(log_u):
        return mu - log1mexp(log_u + log1mexp(eta_mu)) / alpha_mu

    def _sample_from_overlap(log_u):
        # upper bound
        log_d = logsubexp(alpha_mu * mu, alpha_eta * eta - (alpha_eta - alpha_mu) * gamma) - log_tZ
        xp2 = -(1.0 / alpha_eta) * (log_u - log_d)

        def log_f(x):
            return logsubexp(alpha_mu * (x - mu), - alpha_eta * (x - eta) * gamma) - log_tZ - log_u

        res, *_ = tfp.math.find_root_chandrupatla(log_f, gamma, xp2, position_tolerance=1e-6,
                                                  value_tolerance=1e-6)

        return res

    return jax.lax.cond(log_u1 < log_p, _sample_from_overlap, _sample_from_tail, jnp.log(jax.random.uniform(key2)))


def _sample_from_second_marginal(key, mu, eta, alpha_mu, alpha_eta, eta_mu, gamma_mu, gamma_eta, gamma):
    log_Zq = log1mexp(logsubexp(jnp.logaddexp(eta_mu, gamma_mu), gamma_eta))
    log_u = jnp.log(jax.random.uniform(key))

    # upper bound
    log_d = logsubexp(alpha_mu * mu, alpha_eta * eta - (alpha_eta - alpha_mu) * gamma) - log_Zq
    xp2 = -(1.0 / alpha_eta) * (log_u - log_d)

    def log_f(x):
        res = logsubexp(eta_mu, -alpha_mu * (x - mu))
        res = jnp.logaddexp(res, -alpha_eta * (x - eta)) - log_Zq - logsubexp(-log_Zq, log_u)
        return res

    out, objective_at_estimated_root, *_ = tfp.math.find_root_chandrupatla(log_f, gamma, xp2, position_tolerance=1e-6,
                                                                           value_tolerance=1e-6)

    return out


def _robert_sampler(key, mu, alpha):
    def body(carry):
        curr_k, *_ = carry
        curr_k, subkey = jax.random.split(curr_k, 2)

        u1, u2 = jax.random.uniform(subkey, shape=(2,))

        x = mu - jnp.log(1 - u1) / alpha
        accepted = u2 <= jnp.exp(-0.5 * (x - alpha) ** 2)

        return curr_k, x, accepted

    _, x_out, _ = jax.lax.while_loop(lambda carry: ~carry[-1], body, (key, 0., False))
    return x_out


def get_alpha(mu):
    """ Compute the optimal alpha as per Robert (1995) """
    return 0.5 * (mu + jnp.sqrt(mu ** 2 + 4))


def get_gamma(mu, eta, alpha, beta):
    """ Threshold when hatp(x) = hatq(x) """
    return (jnp.log(beta) - jnp.log(alpha) + beta * eta - alpha * mu) / (beta - alpha)


def texp_logpdf(x, mu, alpha):
    """ Translated exponential density """
    return jnp.where(x < mu, -jnp.inf, jnp.log(alpha) - alpha * (x - mu))

