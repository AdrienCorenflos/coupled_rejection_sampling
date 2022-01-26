"""
This corresponds to the Application 2 of [1].
"""
import math
import time
from functools import partial

import jax.numpy as jnp
import jax.random
import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import tqdm.auto as tqdm
from jax.scipy.special import logsumexp

from coupled_rejection_sampling.coupled_parallel_resampler import coupled_parallel_resampler

RUN = False
PLOT = True
save_path = "out/resampling.npz"
M = 2 ** 14
B = 100  # number of times to run the experiment

jax_key = jax.random.PRNGKey(12345)
jax_key, x_key, z_key = jax.random.split(jax_key, 3)

xs = jax.random.normal(x_key, (M,))
zs = jax.random.normal(z_key, (M,))
YS = jnp.arange(0, 4, 1)

NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def experiment():
    log_M = -0.5 * math.log(2 * math.pi)
    keys = jax.random.split(jax_key, B)

    @partial(jax.jit, static_argnums=(0,))
    def experiment_one_n(N, k, log_w_x, log_w_z):
        *_, are_coupled, n_trials = coupled_parallel_resampler(k, log_w_x, log_w_z, log_M, log_M, N)
        return are_coupled, n_trials

    are_coupled_res = np.empty((B, len(NS), len(YS), M), dtype=np.float32)
    n_trials_res = np.empty((B, len(NS), len(YS), M), dtype=np.int16)
    coupling_proba = np.empty((len(YS),))
    runtime = np.empty((B, len(NS), len(YS)), dtype=np.float32)
    for j, y in enumerate(tqdm.tqdm(YS)):
        log_w_X = jstats.norm.logpdf(xs, y)
        log_w_Z = jstats.norm.logpdf(zs, y)

        # normalized
        log_W_X = log_w_X - logsumexp(log_w_X)
        log_W_Z = log_w_Z - logsumexp(log_w_Z)

        log_nu = jnp.minimum(log_W_X, log_W_Z)
        coupling_proba[j] = jnp.exp(logsumexp(log_nu))

        for i, n in enumerate(NS):
            *_, block = experiment_one_n(n, keys[0], log_w_X, log_w_Z)
            block.block_until_ready()

            for b in range(B):
                tic = time.time()
                are_coupled_res_b, n_trials_res_b = experiment_one_n(n, keys[b], log_w_X, log_w_Z)
                are_coupled_res[b, i, j, :], n_trials_res[b, i, j, :] = are_coupled_res_b, n_trials_res_b
                are_coupled_res_b.block_until_ready()  # noqa
                runtime[b, i, j] = (time.time() - tic) / B
    return coupling_proba, are_coupled_res.mean(0), are_coupled_res.std(0), n_trials_res.mean(0), n_trials_res.std(
        0), runtime


if RUN:
    th_coupling_proba, coupling_mean, coupling_std, n_trials_mean, n_trials_std, runtime_res = experiment()
    np.savez(save_path, M=M, NS=NS, YS=YS, B=B, n_trials_mean=n_trials_mean, n_trials_std=n_trials_std,
             coupling_mean=coupling_mean, coupling_std=coupling_std, th_coupling_proba=th_coupling_proba,
             runtime=runtime_res)



if PLOT:
    data = np.load(save_path)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")

    for j, y in enumerate(data["YS"]):
        ax = axes.flatten()[j]
        ax.step(NS, np.mean(data["coupling_mean"][:, j, :], -1), color="tab:blue", where="post")
        ax.fill_between(NS,
                        np.mean(data["coupling_mean"][:, j, :], -1) - 1.96 * np.std(data["coupling_mean"][:, j, :], -1),
                        np.mean(data["coupling_mean"][:, j, :], -1) + 1.96 * np.std(data["coupling_mean"][:, j, :], -1),
                        color="tab:blue", alpha=0.75, step="post")
        ax.set_title(f"$y={y}$")
        ax.axhline(data["th_coupling_proba"][j], 0, 1, label=f"Theoretical", linestyle="--", color="k", linewidth=2.)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
    tikzplotlib.save("out/resampling_coupling_proba.tikz")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), sharex=True, sharey=True)
    for j, y in enumerate(data["YS"]):
        ax = axes.flatten()[j]
        ax.step(NS, np.mean(data["n_trials_mean"][:, j, :], -1), where="post")
        ax.fill_between(NS,
                        np.mean(data["n_trials_mean"][:, j, :], -1) - 1.96 * np.std(data["n_trials_mean"][:, j, :], -1),
                        np.mean(data["n_trials_mean"][:, j, :], -1) + 1.96 * np.std(data["n_trials_mean"][:, j, :], -1),
                        color="tab:blue", alpha=0.75, step="post")
        ax.set_title(f"$y={y}$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

    tikzplotlib.save("out/resampling_ntrials.tikz")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), sharex=True, sharey=True)
    for j, y in enumerate(data["YS"]):
        ax = axes.flatten()[j]
        ax.step(NS, np.mean(data["runtime"][..., j], 0), where="post")
        ax.fill_between(NS,
                        np.mean(data["runtime"][..., j], 0) - 1.96 * np.std(data["runtime"][..., j], 0),
                        np.mean(data["runtime"][..., j], 0) + 1.96 * np.std(data["runtime"][..., j], 0),
                        color="tab:blue", alpha=0.75, step="post")
        ax.set_title(f"$y={y}$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)

    tikzplotlib.save("out/resampling_runtime.tikz")
