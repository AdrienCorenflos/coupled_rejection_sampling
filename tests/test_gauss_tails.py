import jax.random
import jax.lax
from jax import numpy as jnp
import numpy as np
import scipy.integrate
import pytest
import numpy.testing as np_test

from jax.config import config
config.update("jax_enable_x64", True)

import coupled_rejection_sampling.gauss_tails as gauss_tails

@pytest.mark.parametrize("mu", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("eta_mu", [1.0, 2.0])
def test_c_dens(mu, eta_mu):
    eta = mu + eta_mu

    alpha = gauss_tails.get_alpha(mu)
    beta = gauss_tails.get_alpha(eta)

    tp = lambda x: np.where(x < mu, 0.0, alpha * np.exp(-alpha * (x-mu)))
    tq = lambda x: np.where(x < eta, 0.0, beta * np.exp(-beta * (x-eta)))
    tc = lambda x: np.min(np.stack([tp(x),tq(x)]),axis=0)
    Z1 = scipy.integrate.quad(tc, 0, np.Inf)[0]

    gt = gauss_tails.GaussTails(mu, eta)
    Z2 = gt.pxy()

    np_test.assert_allclose(Z1, Z2, atol=1e-6, rtol=1e-6)

    xs = np.linspace(mu,10,1000)
    c_dens1 = tc(xs)/Z1
    c_dens2 = jnp.exp(gt.c_logpdf(xs))
    np_test.assert_allclose(c_dens1, c_dens2, atol=1e-6, rtol=1e-6)

def test_c_sample():
    mu = 1
    eta = 1.2

    alpha = gauss_tails.get_alpha(mu)
    beta = gauss_tails.get_alpha(eta)
    tp = lambda x: np.where(x < mu, 0.0, alpha * np.exp(-alpha * (x-mu)))
    tq = lambda x: np.where(x < eta, 0.0, beta * np.exp(-beta * (x-eta)))
    tc = lambda x: np.min(np.stack([tp(x),tq(x)]),axis=0)
    Z1 = scipy.integrate.quad(tc, 0, np.Inf)[0]
    sinint1 = scipy.integrate.quad(lambda x: np.sin(x) * tc(x) / Z1, 0, np.Inf)[0]
    cosint1 = scipy.integrate.quad(lambda x: np.cos(x) * tc(x) / Z1, 0, np.Inf)[0]

    key = jax.random.PRNGKey(1)
    gt = gauss_tails.GaussTails(mu, eta)
    samples = gt.c_sample(key, 100000)
    sinint2 = np.sin(samples).mean()
    cosint2 = np.cos(samples).mean()

    np_test.assert_allclose(sinint1, sinint2, atol=1e-2, rtol=1e-2)
    np_test.assert_allclose(cosint1, cosint2, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("u", [0.00001, 0.1, 0.5, 0.99999])
def test_tp1_tq_inv(u):
    mu = 1
    eta = 1.2

    gt = gauss_tails.GaussTails(mu, eta)

    tZ = gt.e_alpha_gamma_mu - gt.e_beta_gamma_eta

    x, _, _ = gt.TP1_inv(u)
    err = (jnp.exp(-gt.alpha * (x - gt.mu)) - jnp.exp(-gt.beta * (x - gt.eta))) / tZ - u

    np_test.assert_allclose(err, 0.0, atol=1e-2, rtol=1e-2)
    assert x > gt.gamma

    Zq = 1 - gt.e_alpha_eta_mu + gt.e_alpha_gamma_mu - gt.e_beta_gamma_eta
    x, _, _ = gt.TQ_inv(u)
    err = (1 - gt.e_alpha_eta_mu + jnp.exp(-gt.alpha * (x - gt.mu)) - jnp.exp(-gt.beta * (x - gt.eta))) / Zq - u

    np_test.assert_allclose(err, 0.0, atol=1e-2, rtol=1e-2)
    assert x < gt.gamma
