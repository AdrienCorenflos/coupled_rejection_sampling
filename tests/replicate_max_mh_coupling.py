"""
This is to replicate the Figure 7 in Maximal Couplings of the Metropolis-Hastings.
As discussed with authors, there was a mistake in their code so that P_{MR} works better than P_C.

"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from coupled_rejection_sampling.mvn import reflection_maximal
from coupled_rejection_sampling.reflected_coupled_mh import reflection_coupled_mh

DS = np.arange(1, 21, dtype=int)
M = 1_000
JAX_KEY = jax.random.PRNGKey(42)


@partial(jax.jit, static_argnums=(0, 1))
def experiment(D, kernel):

    @jax.vmap
    def get_meeting_time(key):
        def cond(carry):
            return ~carry[-1]

        def body(carry):
            curr_k, x, y, i, *_ = carry
            curr_k, op_k = jax.random.split(curr_k)
            x, y, coupled = kernel(op_k, x, y)
            return curr_k, x, y, i + 1, coupled

        loop_key, init_key = jax.random.split(key, 2)
        x0, y0 = jax.random.normal(init_key, (2, D))

        *_, coupling_time, _ = jax.lax.while_loop(cond, body, (loop_key, x0, y0, 0, False))
        return coupling_time

    return get_meeting_time(jax.random.split(JAX_KEY, M))


@jax.jit
def status_quo_kernel(key, x, y, chol):
    proposal_key, acceptance_key = jax.random.split(key)
    x_prop, y_prop, coupled_prop = reflection_maximal(proposal_key, 1, x, y, chol)
    x_prop, y_prop, coupled_prop = x_prop[0], y_prop[0], coupled_prop[0]

    log_alpha_x = -0.5 * jnp.sum(x_prop ** 2) + 0.5 * jnp.sum(x ** 2)
    log_alpha_y = -0.5 * jnp.sum(y_prop ** 2) + 0.5 * jnp.sum(y ** 2)

    log_u = jnp.log(jax.random.uniform(acceptance_key))
    accept_x = log_u < log_alpha_x
    accept_y = log_u < log_alpha_y

    x = jax.lax.select(accept_x, x_prop, x)
    y = jax.lax.select(accept_y, y_prop, y)

    coupled = accept_x & accept_y & coupled_prop
    return x, y, coupled


@jax.jit
def reflected_mh_kernel(key, x, y, chol):
    target_logpdf = lambda z: -0.5 * jnp.sum(z ** 2)
    proposal = lambda k, z: z + chol @ jax.random.normal(k, z.shape)
    proposal_logpdf = lambda z, z_prime: 0.  # Symmetric proposal, so doesn't matter
    step = reflection_coupled_mh(proposal, proposal_logpdf, target_logpdf)
    (x, _), (y, _), coupled = step(key, x, y)
    return x, y, coupled


if __name__ == "__main__":
    status_quo_res = np.empty((len(DS), M))
    reflected_res = np.empty((len(DS), M))

    for i, d in enumerate(tqdm.tqdm(DS)):
        proposal_chol = 2.38 * np.eye(d) / d ** 0.5
        status_quo_res[i] = experiment(d, partial(status_quo_kernel, chol=proposal_chol))
        reflected_res[i] = experiment(d, partial(reflected_mh_kernel, chol=proposal_chol))

    plt.plot(DS, np.mean(status_quo_res, -1), label="$P_{SQ}$ with $Q_{MR}$", c="tab:blue")
    plt.fill_between(DS,
                     np.percentile(status_quo_res, 5, -1),
                     np.percentile(status_quo_res, 95, -1),
                     color="tab:blue", alpha=0.5)
    plt.plot(DS, np.mean(reflected_res, -1), label="$P_{MR}$", c="tab:orange")
    plt.fill_between(DS,
                     np.percentile(reflected_res, 5, -1),
                     np.percentile(reflected_res, 95, -1),
                     color="tab:orange", alpha=0.5)
    plt.legend()
    plt.show()
