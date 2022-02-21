""" This algorithm implements Algorithm 4 of Maximal Couplings of the Metropolis-Hastings
Algorithm to compare with our method in the manifold MALA case. It is however coded in a fairly generic manner, so I
felt like putting it in the main code.
The notations essentially follow from the paper notations but is all in log space.
"""
import jax
import jax.numpy as jnp

from coupled_rejection_sampling.utils import logsubexp


def reflection_coupled_mh(proposal, proposal_log_density, target_log_density):

    def _residual(x, y):
        def cond(carry):
            return ~carry[-1]

        def body(carry):
            op_key, y_prop, _ = carry
            next_key, subkey1, subkey2 = jax.random.split(op_key)
            y_prop, accepted_y = kernel(subkey2, y)

            def if_accepted(_):
                log_u = jnp.log(jax.random.uniform(subkey1))
                log_p_y_yp = kernel_log_density(y_prop, y)
                log_p_res_y_yp = tr_xy_fn(y, x, y_prop)
                accept = log_u < log_p_res_y_yp - log_p_y_yp
                return y_prop, accept

            return jax.lax.cond(accepted_y, lambda _: (y_prop, False), if_accepted, None)

    def step(key, x, y):
        subkey1, subkey2, subkey3 = key
        x_prime, accepted_x = kernel(subkey1, x)


def _transport_fn(x, y, x_prime):
    r_curr = jnp.linalg.norm(y - x)
    e = jax.lax.select(r_curr < 1e-8, jnp.zeros_like(x), (y - x) / r_curr)
    x_diff = x_prime - x
    eta = x_diff - 2 * e * e.dot(x_diff)
    out = x + eta
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
        return logsubexp(residual, jnp.minimum(residual, log_f_r(y_prime_t, x, y))

    return log_f_t

