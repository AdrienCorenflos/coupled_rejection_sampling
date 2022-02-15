""" This algorithm implements Algorithm 4 of Maximal Couplings of the Metropolis-Hastings
Algorithm to compare with our method in the manifold MALA case. It is however coded in a fairly generic manner, so I
felt like putting it in the main code.
"""
import jax
import jax.numpy as jnp

from coupled_rejection_sampling.utils import logsubexp


def reflection_coupled_mh(kernel, kernel_log_density):
    def r_xy_fn(x, y, x_prime):
        log_p_x_zp = kernel_log_density(x_prime, x)
        log_p_y_zp = kernel_log_density(x_prime, y)
        return logsubexp(log_p_x_zp, log_p_y_zp)

    def tr_xy_fn(x, y, x_prime):
        y_refl = _transport_fn(x, y, x_prime)
        r_xy_xp = r_xy_fn(x, y, x_prime)
        r_yx_yp = r_xy_fn(y, y, y_refl)
        return logsubexp(r_xy_xp, r_yx_yp)


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
    y = x + eta
    return y