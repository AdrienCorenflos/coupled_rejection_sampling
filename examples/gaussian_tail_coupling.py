"""
Example of coupling of tails of N(0,1) Gaussian distributions.
"""

import jax.random
import jax.lax
from jax import numpy as jnp
import numpy as np
import scipy.integrate
import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import coupled_rejection_sampling.gauss_tails as gauss_tails

from jax.config import config
config.update("jax_enable_x64", True)

RUN = False
PLOT = True

save_path = "out/gauss_tails.npz"

if RUN:
    JAX_KEY = jax.random.PRNGKey(0)

    # Tails >5 is considered "hard" and the closer these are, the worse Thorisson's method is
    mu = 6
    eta = 6.2

    K = 100  # How many times to repeat
    M = 10_000  # Samples to draw
    N_list = [1, 10, 20, 30, 40, 50, 60, 80, 90, 100]  # Ensemble sizes

    # Compute maximal coupling probability (note that this can be inaccurate)
    p = lambda x: jnp.where(x >= mu, 1 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-0.5 * x ** 2) / jstats.norm.cdf(-mu), 0.0)
    q = lambda x: jnp.where(x >= eta, 1 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-0.5 * x ** 2) / jstats.norm.cdf(-eta), 0.0)
    mpq = lambda x: np.min(np.stack([p(x), q(x)]), axis=0)
    pxy = scipy.integrate.quad(mpq, 0, np.Inf)[0]
    print(f"Theoretical P(X=Y) = {pxy}")

    def experiment():
        pxy_list = np.empty((K,len(N_list)))

        curr_key = JAX_KEY
        for k in range(K):
            curr_key, key = jax.random.split(curr_key, 2)
            for n in range(len(N_list)):
                N = N_list[n]
                print(f"{k+1}/{K} running with N = {N} ...")
                gt = gauss_tails.GaussTails(mu, eta)
                keys = jax.random.split(key, M)
                tmp = jax.vmap(lambda k: gt.coupled_gauss_tails_rs(k, N))(keys)

                x_samples = tmp[0]
                y_samples = tmp[1]
                acc = tmp[2]

                pxy = acc.sum() / len(acc)
                pxy_list[k,n] = pxy
                print(f"P(X=Y) = {pxy}")

        return x_samples, y_samples, pxy_list

    x_samples, y_samples, pxy_list = experiment()
    np.savez(save_path, x_samples=x_samples, y_samples=y_samples, pxy_list=pxy_list, M=M, N_list=N_list, pxy=pxy, mu=mu, eta=eta)

if PLOT:
    data = np.load(save_path)

    x_samples = data["x_samples"]
    y_samples = data["y_samples"]
    pxy_list = data["pxy_list"]
    M = data["M"]
    N_list = data["N_list"]
    pxy = data["pxy"]
    mu = data["mu"]
    eta = data["eta"]

    mu = 6
    eta = 6.2

    alpha = gauss_tails.get_alpha(mu)
    beta = gauss_tails.get_alpha(eta)
    gamma = gauss_tails.get_gamma(mu, eta, alpha, beta)

    xs = np.linspace(mu, 10, 1000)

    fig, ax = plt.subplots()
    plt.title("Marginal p(y)")

    xh, xbins = np.histogram(x_samples, bins=100, density=True)
    ax.bar(0.5 * (xbins[:-1] + xbins[1:]), xh, xbins[1] - xbins[0], label='Histogram of x samples')

    p = lambda x: np.where(x < mu, 0.0, alpha * np.exp(-alpha * (x - mu)))
    ax.plot(xs, p(xs), 'r-', label='p(x)')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Marginal q(y)")

    yh, ybins = np.histogram(y_samples, bins=100, density=True)
    ax.bar(0.5 * (ybins[:-1] + ybins[1:]), yh, ybins[1] - ybins[0], label='Histogram of y samples')

    q = lambda x: np.where(x < eta, 0.0, beta * np.exp(-beta * (x - eta)))
    ax.plot(xs, q(xs), 'r-', label='q(x)')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Scatter plot of samples")
    ax.scatter(x_samples, y_samples, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    NS = np.array(N_list)
    fig, ax = plt.subplots()
    plt.title("Coupling probability as function of ensemble size")
    ax.plot(NS,pxy_list.mean(axis=0), label='Coupling probability')
    ax.plot(NS,pxy*np.ones_like(NS),'--', label='Maximal coupling')
    plt.xlabel("Ensemble size N")
    plt.ylabel("Probability")
    plt.ylim([0.25,0.3])
    ax.legend()
    plt.show()
