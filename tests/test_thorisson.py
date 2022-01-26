import jax
import jax.random
import numpy as np
import numpy.testing as np_test
import pytest

import coupled_rejection_sampling.mvn as coupled_mvns
import coupled_rejection_sampling.thorisson as thorisson


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("C", [0.5, 0.75, 0.95])
def test_mvns_same_cov(seed, d, C):
    np.random.seed(seed)
    key = jax.random.PRNGKey(1234)
    N = 100_000

    m = np.random.randn(d)
    mu = m + 1e-1 * np.random.randn(d)

    chol_Q = 1. + np.random.rand(d, d)
    chol_Q[np.triu_indices(d, 1)] = 0.

    p = lambda k: coupled_mvns.mvn_sampler(k, 1, m, chol_Q)[0]
    q = lambda k: coupled_mvns.mvn_sampler(k, 1, mu, chol_Q)[0]

    log_p = lambda x: coupled_mvns.mvn_logpdf(x, m, chol_Q)
    log_q = lambda x: coupled_mvns.mvn_logpdf(x, mu, chol_Q)

    bunch_of_keys = jax.random.split(key, N)
    vmapped_thorisson = jax.vmap(lambda k: thorisson.modified_thorisson(k, p, q, log_p, log_q, C))
    xs, ys, are_coupled, n_trials = vmapped_thorisson(bunch_of_keys)

    np_test.assert_allclose(xs.mean(0), m, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(ys.mean(0), mu, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(xs[are_coupled], ys[are_coupled], atol=1e-2)
    np_test.assert_allclose(np.cov(xs, rowvar=False), chol_Q @ chol_Q.T, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(np.cov(ys, rowvar=False), chol_Q @ chol_Q.T, atol=1e-2, rtol=1e-2)

    assert np.mean(n_trials) > 0
    assert np.mean(n_trials) == pytest.approx(2, rel=1e-2, abs=1e-3)
    assert np.var(n_trials) < 2 * C / (1 - C) + 1e-1  # 1e-1 is just an additional buffer
