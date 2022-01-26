import jax
import jax.numpy as jnp


def modified_thorisson(key, p, q, log_p, log_q, C=1.):
    """
    Modified version of Thorisson coupling algorithm [4] as suggested in [5].

    Parameters
    ----------
    Parameters
    ----------
    key: jnp.ndarray
       JAX random key
    p, q: callable
        Sample from the marginals. Take a JAX key and returns a sample.
    log_p, log_q: callable
        The log densities of the dominating marginals and the target ones. They take arrays (d) and return a float.
    C: float, optional
        Constant to control the variance of the run time, this will be clipped between 0 and 1. Default is 1.

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

    C = jnp.clip(C, 0., 1.)
    key, init_key, init_accept_key = jax.random.split(key, 3)

    log_w = lambda x: log_q(x) - log_p(x)

    def log_phi(x):
        return jnp.minimum(log_w(x), jnp.log(C))

    X = p(init_key)
    log_u = jnp.log(jax.random.uniform(init_accept_key))

    # P(accept) = phi(X)
    accept_X_init = log_u < log_phi(X)

    def cond(carry):
        accepted, *_ = carry
        return ~accepted

    def body(carry):
        *_, i, current_key = carry
        next_key, sample_key, accept_key = jax.random.split(current_key, 3)
        Y = q(sample_key)
        log_v = jnp.log(jax.random.uniform(accept_key))

        # P(accept) = 1 - phi(Y)/w(Y)
        accept = log_v > log_phi(Y) - log_w(Y)
        return accept, Y, i + 1, next_key

    _, Z, n_trials, _ = jax.lax.while_loop(cond, body, (accept_X_init, X, 1, key))

    return X, Z, accept_X_init, n_trials
