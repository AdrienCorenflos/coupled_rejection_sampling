"""
Riemannian manifold MALA logistic regression example. This corresponds to the Application 4 of [1].
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np
import pandas as pd
import tikzplotlib
import tqdm.auto as tqdm
from jax.scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from coupled_rejection_sampling.mvn import coupled_mvns, mvn_logpdf
from coupled_rejection_sampling.thorisson import modified_thorisson

JAX_KEY = jax.random.PRNGKey(0)
B = 1_000  # number of parallel coupled chains
K = 50  # number of experiments
CS = np.linspace(0.8, 0.99, 5)
NS = [16, 32, 64]

ALPHA = 100.  # Same wide prior as in mMALA paper
EPS = 0.05

RUN = True
PLOT = True

data_path = "data/heart.csv"
save_path = "out/mmala.npz"


def thorisson_sample(C):
    def sampler(key, x_mean, x_chol, y_mean, y_chol):
        def p_xy(k, mean, scale):
            return mean + scale @ jax.random.normal(k, (mean.shape[0],))

        X, Y, coupled, _ = modified_thorisson(key,
                                              lambda k: p_xy(k, x_mean, x_chol),
                                              lambda k: p_xy(k, y_mean, y_chol),
                                              lambda theta: mvn_logpdf(theta, x_mean, x_chol),
                                              lambda theta: mvn_logpdf(theta, y_mean, y_chol),
                                              C)
        return X, Y, coupled

    return sampler


def rejection_sample(N):
    def sampler(key, x_mean, x_chol, y_mean, y_chol):
        X, Y, coupled, _ = coupled_mvns(key, x_mean, x_chol, y_mean, y_chol, N)
        return X, Y, coupled

    return sampler


def _get_manifold_langevin_discretisation(theta, eps, log_pi):
    """Gets the mean and cholesky covariance of the proposal. This only considers flat manifolds"""
    d = theta.shape[0]

    # Compute gradient and Fisher information
    value, grad = jax.value_and_grad(log_pi)(theta)
    fisher = -jax.hessian(log_pi)(theta)

    # Get sqrt of inverse Fisher
    chol_fisher = jlinalg.cholesky(fisher, lower=True)
    inv_chol_fisher = jlinalg.solve_triangular(chol_fisher, jnp.eye(d), lower=True)

    # Get mean
    mean = 0.5 * eps ** 2 * jlinalg.cho_solve((chol_fisher, True), grad)

    return mean + theta, eps * inv_chol_fisher


@partial(jax.jit, static_argnums=(4, 5))
def simplified_manifold_mala_step(key, x, y, eps, sampler, log_pi):
    sampling_key, accept_key = jax.random.split(key, 2)

    prop_mean_x, prop_chol_x = _get_manifold_langevin_discretisation(x, eps, log_pi)
    prop_mean_y, prop_chol_y = _get_manifold_langevin_discretisation(y, eps, log_pi)

    x_star, y_star, coupled = sampler(sampling_key, prop_mean_x, prop_chol_x, prop_mean_y, prop_chol_y)

    rev_mean_x, rev_chol_x = _get_manifold_langevin_discretisation(x_star, eps, log_pi)
    rev_mean_y, rev_chol_y = _get_manifold_langevin_discretisation(y_star, eps, log_pi)

    alpha_x = log_pi(x_star) - log_pi(x) + mvn_logpdf(x, rev_mean_x, rev_chol_x) - mvn_logpdf(x_star, prop_mean_x,
                                                                                              prop_chol_x)
    alpha_y = log_pi(y_star) - log_pi(y) + mvn_logpdf(y, rev_mean_y, rev_chol_y) - mvn_logpdf(y_star, prop_mean_y,
                                                                                              prop_chol_y)

    log_u = jnp.log(jax.random.uniform(accept_key))

    accept_x = log_u < alpha_x
    accept_y = log_u < alpha_y

    x = jax.lax.select(accept_x, x_star, x)
    y = jax.lax.select(accept_y, y_star, y)

    return x, y, accept_x & accept_y & coupled


@partial(jax.jit, static_argnums=(2, 3, 4))
def sample_coupled_chain(key, eps, sampler, log_pi, D):
    key, init_x_key, init_y_key = jax.random.split(key, 3)

    x0 = jax.random.normal(init_x_key, (D,))
    y0 = jax.random.normal(init_y_key, (D,))

    def cond(carry):
        return ~carry[-1]

    def body(carry):
        op_key, x, y, iteration, _ = carry
        op_key, sample_key = jax.random.split(op_key, 2)
        x, y, coupled = simplified_manifold_mala_step(sample_key, x, y, eps, sampler, log_pi)
        return op_key, x, y, iteration + 1, coupled

    *_, meeting_time, _ = jax.lax.while_loop(cond, body, (key, x0, y0, 0, False))
    return meeting_time


def experiment():
    df = pd.read_csv(data_path, header=0)
    dim = df.shape[1]

    X = df.values[:, :-1]
    X = X - np.mean(X, 0, keepdims=True)
    X = X / np.std(X, 0, keepdims=True)

    X = np.pad(X, [(0, 0), (0, 1)], constant_values=1)

    y = df.values[:, -1].astype(bool)

    def log_target(theta):
        prior = norm.logpdf(theta, 0., ALPHA ** 0.5).sum()

        temp = -X @ theta
        log_probs_true = -jnp.logaddexp(0, temp)
        log_probs_false = temp + log_probs_true
        log_lik = jnp.where(y, log_probs_true, log_probs_false)
        return prior + jnp.nansum(log_lik)

    rejection_meeting_times_res = np.empty((len(NS), K, B))
    thorisson_meeting_times_res = np.empty((len(CS), K, B))

    rejection_runtime_res = np.empty((len(NS), K))
    thorisson_runtime_res = np.empty((len(CS), K))

    # Rejection:
    rej_key = JAX_KEY
    for i, N in enumerate(tqdm.tqdm(NS, leave=False)):
        rej_sampler = rejection_sample(N)
        rej_experiment_fun = jax.jit(
            jax.vmap(lambda op_key: sample_coupled_chain(op_key, EPS, rej_sampler, log_target, dim)))

        # run it once to compile
        batched_keys = jax.random.split(JAX_KEY, B)
        block = rej_experiment_fun(batched_keys)
        block.block_until_ready()
        for k in enumerate(tqdm.trange(K, leave=False)):
            tic = time.time()
            rej_key, subkey = jax.random.split(rej_key)
            batched_keys = jax.random.split(subkey, B)

            rej_out = rej_experiment_fun(batched_keys)
            rej_out.block_until_ready()
            rejection_meeting_times_res[i, k] = rej_out
            rejection_runtime_res[i, k] = time.time() - tic

    thor_key = JAX_KEY
    for j, C in enumerate(tqdm.tqdm(CS, leave=False)):
        thor_sampler = thorisson_sample(C)
        thor_experiment_fun = jax.jit(
            jax.vmap(lambda op_key: sample_coupled_chain(op_key, EPS, thor_sampler, log_target, dim)))

        # run it once to compile
        batched_keys = jax.random.split(JAX_KEY, B)
        block = thor_experiment_fun(batched_keys)
        block.block_until_ready()
        for k in tqdm.trange(K):
            tic = time.time()
            thor_key, subkey = jax.random.split(thor_key)
            batched_keys = jax.random.split(subkey, B)
            thor_out = thor_experiment_fun(batched_keys)
            thor_out.block_until_ready()
            thorisson_meeting_times_res[j, k] = thor_out
            thorisson_runtime_res[j, k] = time.time() - tic

    return rejection_meeting_times_res, rejection_runtime_res, thorisson_meeting_times_res, thorisson_runtime_res


if RUN:
    rejection_meeting_times, rejection_runtime, thorisson_meeting_times, thorisson_runtime = experiment()
    np.savez(save_path, CS=CS, NS=NS, rejection_meeting_times=rejection_meeting_times,
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
