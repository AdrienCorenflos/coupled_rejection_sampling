import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import numpy.testing as np_test
import pytest

from coupled_rejection_sampling.coupled_parallel_resampler import coupled_parallel_resampler


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_coupled_parallel_resampler(seed):
    np.random.seed(42)
    key = jax.random.PRNGKey(1234)
    N = 10_000
    n_particles = 50

    w_X = np.random.rand(n_particles)
    w_Y = np.random.rand(n_particles)

    nu = np.minimum(w_X / w_X.sum(), w_Y / w_Y.sum())
    alpha = np.sum(nu)

    log_w_X = np.log(w_X)
    log_w_Y = np.log(w_Y)

    log_M_X = 0.
    log_M_Y = 0.

    keys = jax.random.split(key, N)

    X_idx, Y_idx, is_coupled, n_trials = jax.vmap(
        lambda k: coupled_parallel_resampler(k, log_w_X, log_w_Y, log_M_X, log_M_Y, 10))(keys)

    bins_X = jax.vmap(lambda x: jnp.bincount(x, length=n_particles))(X_idx)
    bins_Y = jax.vmap(lambda x: jnp.bincount(x, length=n_particles))(Y_idx)
    np_test.assert_allclose(w_X / w_X.sum(), jnp.mean(bins_X, 0) / n_particles, rtol=1e-1, atol=1e-3)
    np_test.assert_allclose(w_Y / w_Y.sum(), jnp.mean(bins_Y, 0) / n_particles, rtol=1e-1, atol=1e-3)

    np_test.assert_allclose(X_idx[is_coupled], Y_idx[is_coupled])
    np_test.assert_allclose(X_idx[is_coupled], Y_idx[is_coupled])

    np_test.assert_allclose(np.mean(is_coupled), alpha, rtol=1e-2, atol=1e-2)
    assert np.mean(is_coupled) <= alpha
