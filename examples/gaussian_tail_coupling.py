"""
Example of coupling of tails of N(0,1) Gaussian distributions.
"""

import jax.lax
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from jax.config import config
from scipy.stats import truncnorm

from coupled_rejection_sampling.gauss_tails import coupled_gaussian_tails, get_alpha, get_gamma

config.update("jax_enable_x64", True)

RUN = True
PLOT = True

save_path = "out/gauss_tails.npz"

if RUN:
    JAX_KEY = jax.random.PRNGKey(0)

    # Tails >5 is considered "hard" and the closer these are, the worse Thorisson's method is
    mu = 6
    eta = 6.1

    M = 100_000  # Samples to draw
    N_list = np.arange(1, 20, dtype=int)  # Ensemble sizes

    # Compute maximal coupling probability (note that this can be inaccurate)
    p = lambda x: truncnorm.pdf(x, mu, np.inf)
    q = lambda x: truncnorm.pdf(x, eta, np.inf)
    mpq = lambda x: np.minimum(p(x), q(x))
    true_pxy = scipy.integrate.quad(mpq, 0, np.Inf)[0]
    print(f"Theoretical P(X=Y) = {true_pxy}")


    def experiment():
        pxy_list = np.empty((len(N_list)))
        x_samples = np.empty((len(N_list), M))
        y_samples = np.empty((len(N_list), M))

        for n in range(len(N_list)):
            N = N_list[n]
            keys = jax.random.split(JAX_KEY, M)
            x_samples[n], y_samples[n], acc, _ = jax.vmap(lambda k: coupled_gaussian_tails(k, mu, eta, N))(
                keys)
            pxy = np.mean(acc)
            pxy_list[n] = pxy
            print(f"iteration = {n}, P(X=Y) = {pxy}")

        return x_samples, y_samples, pxy_list


    x_samples, y_samples, pxy_list = experiment()
    np.savez(save_path, x_samples=x_samples, y_samples=y_samples, pxy_list=pxy_list, M=M, true_pxy=true_pxy, N_list=N_list, mu=mu,
             eta=eta)

if PLOT:
    data = np.load(save_path)

    x_samples = data["x_samples"]
    y_samples = data["y_samples"]
    pxy_list = data["pxy_list"]
    M = data["M"]
    N_list = data["N_list"]
    true_pxy = data["true_pxy"]
    mu = data["mu"]
    eta = data["eta"]

    mu = data["mu"]
    eta = data["eta"]

    alpha = get_alpha(mu)
    beta = get_alpha(eta)
    gamma = get_gamma(mu, eta, alpha, beta)

    xs = np.linspace(mu, 10, 1000)

    fig, ax = plt.subplots()
    plt.title("Marginal p(y)")

    xh, xbins = np.histogram(x_samples[0], bins=100, density=True)
    ax.bar(0.5 * (xbins[:-1] + xbins[1:]), xh, xbins[1] - xbins[0], label='Histogram of x samples')

    p = lambda x: np.where(x < mu, 0.0, alpha * np.exp(-alpha * (x - mu)))
    ax.plot(xs, p(xs), 'r-', label='p(x)')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Marginal q(y)")

    yh, ybins = np.histogram(y_samples[0], bins=100, density=True)
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
    ax.plot(NS, pxy_list, label='Coupling probability')
    ax.hlines(true_pxy, 0, 1, transform=ax.get_yaxis_transform(), linestyles='--', label='Maximal coupling')
    plt.xlabel("Ensemble size N")
    plt.ylabel("Probability")
    plt.ylim([true_pxy - 0.01, true_pxy + 0.01])
    plt.xlim([min(NS), max(NS)])
    ax.legend()
    plt.show()
