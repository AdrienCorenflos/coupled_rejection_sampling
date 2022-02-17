# Coupled sampling from 2 different tails x > mu and x > eta of N(x | 0,1) Gaussian
# by using a maximal coupling of translated exponential proposals hatp(x) = alpha exp(-alpha (x - mu))
# and hatq(x) = beta exp(-beta (x - eta)) similarly to Robert (1995), but with the coupling.
#
# Robert, C. P. (1995). Simulation of truncated normal variables. Statistics and Computing, Volume 5, pages 121–125.
import chex as chex
import jax.lax
import jax.numpy as jnp
import jax.random
import tensorflow_probability.substrates.jax as tfp

from coupled_rejection_sampling.utils import logsubexp, log1mexp


@jax.jit
def coupled_gaussian_tails(key: chex.PRNGKey,
                           mu: chex.Numeric, eta: chex.Numeric):
    """
    A coupled version of Robert's truncated normal sampling algorithm.
    We want to sample from 2 different tails x > mu and x > eta of a N(x | 0,1) Gaussian

    Parameters
    ----------
    key: jnp.ndarray
        JAX random key
    mu, eta: float
        The tails of the Gaussians we want to sample from

    Returns
    -------
    X: chex.Numeric
        The sample from x > mu
    Y: chex.Numeric
        The sample from x > eta
    is_coupled: jnp.ndarray
        Coupling flag
    """
    alpha_mu = get_alpha(mu)
    alpha_eta = get_alpha(eta)

    p = lambda k: _robert_sampler(k, mu, alpha_mu)
    q = lambda k: _robert_sampler(k, eta, alpha_eta)

    log_w_p = lambda x: -0.5 * (x - alpha_mu) ** 2
    log_w_q = lambda x: -0.5 * (x - alpha_eta) ** 2

    def cond(carry):
        accept_X, accept_Y, *_ = carry
        return ~accept_X & ~accept_Y

    def body(carry):
        *_, i, curr_key = carry
        next_key, sample_key, accept_key = jax.random.split(curr_key, 3)
        X_hat, Y_hat, are_coupled = coupled_exponentials(sample_key, mu, alpha_mu, eta, alpha_eta)
        log_w_X = log_w_p(X_hat)
        log_w_Y = log_w_q(Y_hat)

        log_u = jnp.log(jax.random.uniform(accept_key))
        accept_X = log_u < log_w_X
        accept_Y = log_u < log_w_Y

        return accept_X, accept_Y, X_hat, Y_hat, are_coupled, i + 1, next_key

    # initialisation
    residual_key, loop_key = jax.random.split(key)

    output = jax.lax.while_loop(cond,
                                lambda carry: body(carry),
                                (False, False, 0., 0., False, 0, loop_key))

    is_X_accepted, is_Y_accepted, X, Y, is_coupled, n_trials, _ = output

    X = jax.lax.cond(is_X_accepted, lambda _: X, p, residual_key)
    Y = jax.lax.cond(is_Y_accepted, lambda _: Y, q, residual_key)

    is_coupled = is_coupled & is_X_accepted & is_Y_accepted

    return X, Y, is_coupled


@jax.jit
def coupled_exponentials(key:chex.PRNGKey, mu:chex.Numeric, alpha_mu:chex.Numeric, eta:chex.Numeric, alpha_eta:chex.Numeric):
    """
    Sampling from a maximal coupling of shifted exponentials.
    p(x) = exp(-alpha (x - m)) / m  for x >= m, 0 otherwise

    It assumes that eta > mu and alpha_eta > alpha_mu

    Parameters
    ----------
    key: chex.PRNGKey
        JAX random key
    mu, eta: chex.Numeric
        The shift of the exponentials
    alpha_mu, alpha_eta: chex.Numeric
        The rate parameters

    Returns
    -------
    X: chex.Numeric
        The sample from x > mu
    Y: chex.Numeric
        The sample from x > eta
    is_coupled: chex.Numeric
        Coupling flag
    """

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

    x_out, y_out = jax.lax.cond(are_coupled, if_coupled, otherwise, subkey2)
    return x_out, y_out, are_coupled


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

    def log_f(x):
        res = logsubexp(eta_mu, -alpha_mu * (x - mu))
        res = jnp.logaddexp(res, -alpha_eta * (x - eta)) - log_Zq - logsubexp(-log_Zq, log_u)
        return res

    out, objective_at_estimated_root, *_ = tfp.math.find_root_chandrupatla(log_f, eta, gamma, position_tolerance=1e-6,
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
