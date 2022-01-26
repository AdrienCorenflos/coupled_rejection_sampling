"""
This corresponds to the Application 3 of [1].
"""
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm.auto import tqdm

from coupled_rejection_sampling.mvn import coupled_mvns, mvn_logpdf
from coupled_rejection_sampling.thorisson import modified_thorisson

JAX_KEY = jax.random.PRNGKey(0)
B = 10_000  # number of parallel coupled chains
K = 50  # number of experiments
DS = [1, 5, 10]
CS = np.linspace(0.1, 0.99, 20)
NS = [1, 2, 4, 8, 16]

RUN = False
PLOT = True

save_path = "out/gibbs_multidim.npz"


def thorisson_sample(C, D):
    def sample_xy(key, y, y_prime):
        den = 1 + jnp.sum(y ** 2)
        x_mean = jnp.zeros((D,))
        x_chol = (1 / den ** 0.5) * jnp.eye(D)

        den_prime = 1 + jnp.sum(y_prime ** 2)
        x_prime_mean = jnp.zeros((D,))
        x_prime_chol = (1 / den_prime ** 0.5) * jnp.eye(D)

        def p_xy(k, mean, scale):
            return mean + scale @ jax.random.normal(k, (D,))

        X, X_prime, coupled, _ = modified_thorisson(key,
                                                    lambda k: p_xy(k, x_mean, x_chol),
                                                    lambda k: p_xy(k, x_prime_mean, x_prime_chol),
                                                    lambda x: mvn_logpdf(x, x_mean, x_chol),
                                                    lambda x: mvn_logpdf(x, x_prime_mean, x_prime_chol),
                                                    C)
        return X, X_prime, coupled

    return sample_xy, sample_xy


def rejection_sample(N, D):
    def sample_xy(key, y, y_prime):
        den = 1 + jnp.sum(y ** 2)
        x_mean = jnp.zeros((D,))
        x_chol = (1 / den ** 0.5) * jnp.eye(D)

        den_prime = 1 + jnp.sum(y_prime ** 2)
        x_prime_mean = jnp.zeros((D,))
        x_prime_chol = (1 / den_prime ** 0.5) * jnp.eye(D)

        X, X_prime, coupled, _ = coupled_mvns(key, x_mean, x_chol, x_prime_mean, x_prime_chol, N,
                                              jnp.maximum(x_chol, x_prime_chol))
        return X, X_prime, coupled

    return sample_xy, sample_xy


@partial(jax.jit, static_argnums=(1, 2, 3))
def sample_coupled_chain(key, sampler_xy, sampler_yx, D):
    key, init_key = jax.random.split(key, 2)
    x0, x0_prime, y0, y0_prime = jax.random.normal(init_key, (4, D))

    def cond(carry):
        return ~carry[-1]

    def body(carry):
        op_key, x, x_prime, y, y_prime, i, _ = carry
        op_key, x_key, y_key = jax.random.split(op_key, 3)
        x, x_prime, x_coupled = sampler_xy(x_key, y, y_prime)
        y, y_prime, y_coupled = sampler_yx(y_key, x, x_prime)
        return op_key, x, x_prime, y, y_prime, i + 1, x_coupled & y_coupled

    *_, meeting_time, _ = jax.lax.while_loop(cond, body, (key, x0, x0_prime, y0, y0_prime, 0, False))
    return meeting_time


def experiment():
    rejection_meeting_times_res = np.empty((len(DS), len(NS), K, B))
    thorisson_meeting_times_res = np.empty((len(DS), len(CS), K, B))

    rejection_runtime_res = np.empty((len(DS), len(NS), K))
    thorisson_runtime_res = np.empty((len(DS), len(CS), K))

    # Rejection:
    for d, D in enumerate(tqdm(DS)):
        rej_key = JAX_KEY
        for i, N in enumerate(tqdm(NS, leave=False)):
            rej_sample_xy, rej_sample_yx = rejection_sample(N, D)
            rej_experiment_fun = jax.jit(
                jax.vmap(lambda op_key: sample_coupled_chain(op_key, rej_sample_xy, rej_sample_yx, D)))

            # run it once to compile
            batched_keys = jax.random.split(JAX_KEY, B)
            block = rej_experiment_fun(batched_keys)
            block.block_until_ready()
            for k in range(K):
                tic = time.time()
                rej_key, subkey = jax.random.split(rej_key)
                batched_keys = jax.random.split(subkey, B)

                rej_out = rej_experiment_fun(batched_keys)
                rej_out.block_until_ready()
                rejection_meeting_times_res[d, i, k] = rej_out
                rejection_runtime_res[d, i, k] = time.time() - tic

        thor_key = JAX_KEY
        for j, C in enumerate(tqdm(CS, leave=False)):
            thor_sample_xy, thor_sample_yx = thorisson_sample(C, D)
            thor_experiment_fun = jax.jit(
                jax.vmap(lambda op_key: sample_coupled_chain(op_key, thor_sample_xy, thor_sample_yx, D)))

            # run it once to compile
            batched_keys = jax.random.split(JAX_KEY, B)
            block = thor_experiment_fun(batched_keys)
            block.block_until_ready()
            for k in range(K):
                tic = time.time()
                thor_key, subkey = jax.random.split(thor_key)
                batched_keys = jax.random.split(subkey, B)
                thor_out = thor_experiment_fun(batched_keys)
                thor_out.block_until_ready()
                thorisson_meeting_times_res[d, j, k] = thor_out
                thorisson_runtime_res[d, j, k] = time.time() - tic

    return rejection_meeting_times_res, rejection_runtime_res, thorisson_meeting_times_res, thorisson_runtime_res


if RUN:
    rejection_meeting_times, rejection_runtime, thorisson_meeting_times, thorisson_runtime = experiment()
    np.savez(save_path, DS=DS, CS=CS, NS=NS, rejection_meeting_times=rejection_meeting_times,
             rejection_runtime=rejection_runtime,
             thorisson_meeting_times=thorisson_meeting_times, thorisson_runtime=thorisson_runtime)

if PLOT:
    cmap = plt.get_cmap("tab10")
    data = np.load(save_path)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 6), sharey=True)

    for i, D in enumerate(DS):
        axes[0].set_title("Rejection")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].plot(data["NS"], data["rejection_meeting_times"][i].mean(-1).mean(-1),
                     color=cmap(i), label=f"$D={D}$")
        axes[0].fill_between(data["NS"],
                     data["rejection_meeting_times"][i].mean(-1).mean(-1) - 1.96 *
                     data["rejection_meeting_times"][i].mean(-1).std(-1),
                     data["rejection_meeting_times"][i].mean(-1).mean(-1) + 1.96 *
                     data["rejection_meeting_times"][i].mean(-1).std(-1),
                     color=cmap(i), alpha=0.66)

        axes[1].set_title("Thorisson")
        axes[1].plot(data["CS"], data["thorisson_meeting_times"][i].mean(-1).mean(-1),
                     color=cmap(i), label=f"$D={D}$")
        axes[1].fill_between(data["CS"],
                             data["thorisson_meeting_times"][i].mean(-1).mean(-1) - 1.96 *
                             data["thorisson_meeting_times"][i].mean(-1).std(-1),
                             data["thorisson_meeting_times"][i].mean(-1).mean(-1) + 1.96 *
                             data["thorisson_meeting_times"][i].mean(-1).std(-1),
                             color=cmap(i), alpha=0.66)
        axes[1].set_yscale("log")
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[1].legend()
    tikzplotlib.save("out/gibbs_meeting_time.tikz")


    fig, axes = plt.subplots(ncols=2, figsize=(15, 6), sharey=True)

    for i, D in enumerate(DS):
        axes[0].set_title("Rejection")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].plot(data["NS"], data["rejection_runtime"][i].mean(-1),
                     color=cmap(i), label=f"$D={D}$")
        axes[0].fill_between(data["NS"],
                             data["rejection_runtime"][i].mean(-1) - 1.96 *
                             data["rejection_runtime"][i].std(-1),
                             data["rejection_runtime"][i].mean(-1) + 1.96 *
                             data["rejection_runtime"][i].std(-1),
                             color=cmap(i), alpha=0.66)
        axes[1].set_title("Thorisson")
        axes[1].plot(data["CS"], data["thorisson_runtime"][i].mean(-1),
                     color=cmap(i), label=f"$D={D}$")
        axes[1].fill_between(data["CS"],
                             data["thorisson_runtime"][i].mean(-1) - 1.96 *
                             data["thorisson_runtime"][i].std(-1),
                             data["thorisson_runtime"][i].mean(-1) + 1.96 *
                             data["thorisson_runtime"][i].std(-1),
                             color=cmap(i), alpha=0.66)
        axes[1].set_yscale("log")
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[1].legend()

    tikzplotlib.save("out/gibbs_run_time.tikz")
