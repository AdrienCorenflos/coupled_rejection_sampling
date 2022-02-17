import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as np_test
import pytest
import scipy.integrate
from scipy.stats import ks_1samp, truncnorm

import coupled_rejection_sampling.gauss_tails as gauss_tails


@pytest.mark.parametrize("mu", [2., 5., 7.5])
@pytest.mark.parametrize("delta", [1e-5, 1e-2, 1e-1, 1.])
def test_coupled_exponentials(mu, delta):
    N = 500_000
    eta = mu + delta
    alpha_mu = gauss_tails.get_alpha(mu)
    alpha_eta = gauss_tails.get_alpha(eta)

    vmapped_sampler = jax.jit(jax.vmap(lambda k: gauss_tails.coupled_exponentials(k, mu, alpha_mu, eta, alpha_eta)))

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, N)
    xs, ys, coupled = vmapped_sampler(keys)

    theoretical_coupled_proba = scipy.integrate.quad(
        lambda x: np.exp(
            np.minimum(gauss_tails.texp_logpdf(x, mu, alpha_mu), gauss_tails.texp_logpdf(x, eta, alpha_eta))),
        -np.inf,
        np.inf)[0]

    np.testing.assert_allclose(np.mean(coupled), theoretical_coupled_proba, atol=1e-3, rtol=1e-3)

    @jax.jit
    def shifted_exp_cdf(x, m, alpha):
        return jnp.where(x < mu, 0., 1 - jnp.exp(-alpha * (x - m)))

    assert ks_1samp(xs, shifted_exp_cdf, args=(mu, alpha_mu))[0] < 1e-2
    assert ks_1samp(ys, shifted_exp_cdf, args=(eta, alpha_eta))[0] < 1e-2


@pytest.mark.parametrize("mu", [2., 5., 6.5])
@pytest.mark.parametrize("delta", [1e-5, 1e-2, 1e-1, 1.])
def test_coupled_truncated_gaussians(mu, delta):
    N = 500_000
    eta = mu + delta

    vmapped_sampler = jax.jit(jax.vmap(lambda k: gauss_tails.coupled_gaussian_tails(k, mu, eta)))

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, N)
    xs, ys, coupled = vmapped_sampler(keys)

    theoretical_coupled_proba = scipy.integrate.quad(
        lambda x: np.minimum(truncnorm.pdf(x, mu, np.inf), truncnorm.pdf(x, eta, np.inf)),
        -np.inf,
        np.inf)[0]

    np.testing.assert_allclose(np.mean(coupled), theoretical_coupled_proba, atol=1e-1, rtol=1e-1)

    assert ks_1samp(xs, truncnorm.cdf, args=(mu, np.inf))[0] < 1e-2
    assert ks_1samp(ys, truncnorm.cdf, args=(eta, np.inf))[0] < 1e-2
