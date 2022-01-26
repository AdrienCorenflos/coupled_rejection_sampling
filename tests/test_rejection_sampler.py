from functools import partial

import jax.random
import numpy as np
import numpy.testing as np_test
import pytest

import coupled_rejection_sampling.mvn as coupled_mvns
from coupled_rejection_sampling.rejection_sampler import rejection_sampler


@pytest.mark.parametrize("d", [3, 4, 5])
def test_mvns(d):
    np.random.seed(42)
    key = jax.random.PRNGKey(1234)
    N = 100_000

    m = np.random.randn(d)

    chol_P = np.eye(d) + 0.1 * np.random.randn(d, d)
    chol_P[np.triu_indices(d, 1)] = 0.
    chol_Q = 1.5 * np.eye(d)
    log_M = coupled_mvns.tril_log_det(chol_Q) - coupled_mvns.tril_log_det(chol_P)

    log_p = lambda x: coupled_mvns.mvn_logpdf(x, m, chol_P)
    log_p_hat = lambda x: coupled_mvns.mvn_logpdf(x, m, chol_Q)
    p_hat = partial(coupled_mvns.mvn_sampler, m=m, chol_P=chol_Q)

    keys = jax.random.split(key, N)
    xs, n_trials = jax.vmap(lambda k: rejection_sampler(k, p_hat, log_p_hat, log_p, log_M))(keys)

    np_test.assert_allclose(xs.mean(0), m, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(np.cov(xs, rowvar=False), chol_P @ chol_P.T, atol=1e-2, rtol=1e-2)
    assert np.mean(n_trials) == pytest.approx(np.exp(log_M), rel=1e-2, abs=1e-2)
