""" This algorithm implements Algorithm 4 of Maximal Couplings of the Metropolis-Hastings
Algorithm to compare with our method in the manifold MALA case. It is however coded in a fairly generic manner, so I
felt like putting it in the main code.
The notations essentially follow from the paper notations but is all in log space.
"""
import jax
import jax.numpy as jnp

from coupled_rejection_sampling.utils import logsubexp


def reflection_coupled_mh(proposal, proposal_log_density, target_log_density):
    log_f = _get_log_f(proposal_log_density, target_log_density)
    log_f_r = _get_log_f_r(log_f)
    log_f_t = _get_log_f_t(log_f)

    def log_acceptance_ratio(x, x_prime):
        log_q_x_x_prime = proposal_log_density(x, x_prime)
        log_q_x_prime_x = proposal_log_density(x_prime, x)

        log_pi_x = target_log_density(x)
        log_pi_x_prime = target_log_density(x_prime)
        return jnp.minimum(0., log_pi_x_prime - log_pi_x + log_q_x_prime_x - log_q_x_x_prime)

    def mh_step(key, x):
        proposal_key, acceptance_key = jax.random.split(key)
        x_prime = proposal(proposal_key, x)
        log_alpha = log_acceptance_ratio(x, x_prime)
        log_u = jnp.log(jax.random.uniform(acceptance_key))
        return x_prime, log_u < log_alpha

    def step_y(key, x, y, x_prime, accepted_x):
        y_prime = _transport_fn(x, y, x_prime)
        reflection_key, residual_key = jax.random.split(key)
        log_u = jnp.log(jax.random.uniform(reflection_key))
        cond = accepted_x & (log_u + log_f_r(x_prime, x, y) < log_f_r(y_prime, y, x))

        return jax.lax.cond(cond, lambda _: (y_prime, True), lambda k: residual_y(k, y, x), residual_key)

    def residual_y(key, y, x):
        def cond(carry):
            return ~carry[-1]

        def body(carry):
            curr_key, *_ = carry
            next_key, mh_key, accept_key = jax.random.split(curr_key, 3)
            y_prime, accepted_y = mh_step(mh_key, y)
            log_u = jnp.log(jax.random.uniform(accept_key))
            stop = (~accepted_y) | (log_u + log_f(y, y_prime) < log_f_t(y_prime, y, x))
            return next_key, y_prime, accepted_y, stop

        out = jax.lax.while_loop(cond, body, (key, y, False, False))
        return out[1], out[2]

    def step(key, x, y):
        x_key, coupling_key, residual_key = jax.random.split(key, 3)
        x_prime, accepted_x = mh_step(x_key, x)
        log_u = jnp.log(jax.random.uniform(coupling_key))
        next_x = jax.lax.select(accepted_x, x_prime, x)
        cond = accepted_x & (log_u + log_f(x, next_x) < log_f(y, next_x))

        y_prime, accepted_y = jax.lax.cond(cond,
                                           lambda *_: (next_x, accepted_x),
                                           lambda k: step_y(k, x, y, next_x, accepted_x),
                                           residual_key)

        next_y = jax.lax.select(accepted_y, y_prime, y)
        return (next_x, accepted_x), (next_y, accepted_y), cond

    return step


def _transport_fn(x, y, x_prime):
    shape = x.shape
    flat_x = x.flatten()
    flat_y = y.flatten()
    flat_x_prime = x_prime.flatten()
    flat_out = _transport_flat_x(flat_x, flat_y, flat_x_prime)
    out = flat_out.reshape(shape)
    return out


def _transport_flat_x(x, y, x_prime):
    r_curr = jnp.linalg.norm(y - x)
    e = jax.lax.select(r_curr < 1e-8, jnp.zeros_like(x), (y - x) / r_curr)
    x_diff = x_prime - x
    eta = x_diff - 2 * e * e.dot(x_diff)
    out = y + eta
    return out


def _get_log_f(proposal_log_density, target_log_density):
    def log_f(x, x_prime):
        log_q_x_x_prime = proposal_log_density(x, x_prime)
        log_q_x_prime_x = proposal_log_density(x_prime, x)

        log_pi_x = target_log_density(x)
        log_pi_x_prime = target_log_density(x_prime)

        return jnp.minimum(log_q_x_x_prime, log_pi_x_prime + log_q_x_prime_x - log_pi_x)

    return log_f


def _get_log_f_m(log_f):
    def log_f_m(z, x, y):
        return jnp.minimum(log_f(x, z), log_f(y, z))

    return log_f_m


def _get_log_f_r(log_f):
    log_f_m = _get_log_f_m(log_f)

    def log_f_r(x_prime, x, y):
        return logsubexp(log_f(x, x_prime), log_f_m(x_prime, x, y))

    return log_f_r


def _get_log_f_t(log_f):
    log_f_r = _get_log_f_r(log_f)

    def log_f_t(y_prime, y, x):
        residual = log_f_r(y_prime, y, x)
        y_prime_t = _transport_fn(y, x, y_prime)
        return logsubexp(residual, jnp.minimum(residual, log_f_r(y_prime_t, x, y)))

    return log_f_t
