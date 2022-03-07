"""
Example of coupling of tails of N(0,1) Gaussian distributions.
"""
import time

import jax.lax
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import tikzplotlib
import tqdm
from scipy.stats import truncnorm
import seaborn as sns
from coupled_rejection_sampling.gauss_tails import coupled_gaussian_tails

RUN = False
PLOT = True

save_path = "out/gauss_tails.npz"


if RUN:
    JAX_KEY = jax.random.PRNGKey(0)

    # Tails >5 is considered "hard" and the closer these are, the worse Thorisson's method is
    mu = 6.

    M = 100_000  # Samples to draw
    DELTAS = np.linspace(1e-6, 0.5, num=200)

    # Compute maximal coupling probability (note that this can be inaccurate)
    p = lambda x: truncnorm.pdf(x, mu, np.inf)

    def experiment():
        pxy_list = np.empty((len(DELTAS), 2))
        x_samples = np.empty((len(DELTAS), M))
        y_samples = np.empty((len(DELTAS), M))
        runtimes = np.empty((len(DELTAS),))

        keys = jax.random.split(JAX_KEY, M)
        sampler = jax.jit(jax.vmap(coupled_gaussian_tails, in_axes=[0, None, None]))
        # compilation run
        *_, acc = sampler(keys, mu, mu + DELTAS[0])
        acc.block_until_ready()

        for n in tqdm.trange(len(DELTAS)):
            delta = DELTAS[n]
            eta = mu + delta

            q = lambda x: truncnorm.pdf(x, eta, np.inf)
            mpq = lambda x: np.minimum(p(x), q(x))
            true_pxy = scipy.integrate.quad(mpq, 0, np.inf)[0]

            tic = time.time()
            x_samples[n], y_samples[n], acc = sampler(keys, mu, eta)
            acc.block_until_ready()
            toc = time.time()
            runtimes[n] = toc - tic
            pxy = np.mean(acc)
            pxy_list[n] = pxy, true_pxy

        return x_samples, y_samples, pxy_list, runtimes


    x_samples, y_samples, pxy_list, runtimes = experiment()
    np.savez(save_path, x_samples=x_samples, y_samples=y_samples, pxy_list=pxy_list, M=M, etas=mu + DELTAS, mu=mu, runtimes=runtimes)

if PLOT:
    data = np.load(save_path)

    x_samples = data["x_samples"]
    y_samples = data["y_samples"]
    pxy_list = data["pxy_list"]
    M = data["M"]

    mu = data["mu"]
    etas = data["etas"]
    runtimes = data["runtimes"]
    xs = np.linspace(mu, 10, 1000)

    fig = plt.figure(figsize=(10, 10))
    g = sns.jointplot(x=x_samples[10], y=y_samples[10], s=10)

    plt.show()

    fig, ax = plt.subplots()
    plt.title("Coupling probability as function of $\eta$")
    ax.semilogy(etas, pxy_list[:, 0], label='Actual coupling probability', linestyle="-", color="k")
    ax.semilogy(etas, pxy_list[:, 1], label='True coupling probability', linestyle="--", color="k")
    plt.xlabel("$\eta$")
    ax.legend()
    # plt.show()
    tikzplotlib.save("out/gaussian_tails_coupling.tikz")


    fig, ax = plt.subplots()
    plt.title("Run time as function of $\eta$")
    ax.plot(etas, runtimes, linestyle="-", color="k")
    plt.xlabel("$\eta$")
    plt.ylabel("Run time (s)")
    ax.legend()
    tikzplotlib.save("out/gaussian_tails_runtime.tikz")
