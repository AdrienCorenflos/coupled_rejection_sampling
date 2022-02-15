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
import tqdm.auto as tqdm
from jax.scipy.stats import norm
from matplotlib import pyplot as plt

from coupled_rejection_sampling.mvn import coupled_mvns, mvn_logpdf
from coupled_rejection_sampling.thorisson import modified_thorisson

JAX_KEY = jax.random.PRNGKey(0)
K = 10_000  # number of experiments
CS = np.linspace(0.8, 0.99, 7)
NS = [4, 8, 16, 32, 64, 128, 256]

ALPHA = 100.  # Same wide prior as in mMALA paper
EPS = 1.

RUN = False
PLOT = True

data_path = "data/heart.csv"
save_path = "out/mmala.npz"


def thorisson_sample(C):
    @jax.jit
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
    @jax.jit
    def sampler(key, x_mean, x_chol, y_mean, y_chol):
        X, Y, coupled, _ = coupled_mvns(key, x_mean, x_chol, y_mean, y_chol, N)
        return X, Y, coupled

    return sampler


@partial(jax.jit, static_argnums=(2,))
def _get_manifold_langevin_discretisation(theta, eps, log_pi):
    """Gets the mean and cholesky covariance of the proposal. This only considers flat manifolds"""
    d = theta.shape[0]

    # Compute gradient and Fisher information
    grad = jax.grad(log_pi)(theta)
    fisher = -jax.hessian(log_pi)(theta)
    # id_print(fisher, what="fisher autograd")
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

    x0 = 0.1 * jax.random.normal(init_x_key, (D,))
    y0 = 0.1 * jax.random.normal(init_y_key, (D,))

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

    y = df.values[:, -1].astype(float)

    @partial(jax.jit, static_argnames=("return_fisher",))
    def log_target(theta, return_fisher=False):
        prior = norm.logpdf(theta, 0., ALPHA ** 0.5).sum()
        temp = X @ theta
        log_lik = jnp.dot(y, temp) - jnp.sum(jax.nn.softplus(temp))

        if not return_fisher:
            return log_lik + prior

    rejection_meeting_times_res = np.empty((len(NS), K))
    thorisson_meeting_times_res = np.empty((len(CS), K))

    rejection_runtime_res = np.empty((len(NS), K))
    thorisson_runtime_res = np.empty((len(CS), K))

    rej_key = JAX_KEY
    for i, N in enumerate(tqdm.tqdm(NS, leave=True)):
        rej_sampler = rejection_sample(N)
        rej_experiment_fun = jax.jit(lambda op_key: sample_coupled_chain(op_key, EPS, rej_sampler, log_target, dim))

        # run it once to compile
        block = rej_experiment_fun(rej_key)
        block.block_until_ready()
        for k in range(K):
            tic = time.time()
            rej_key, subkey = jax.random.split(rej_key)

            rej_out = rej_experiment_fun(rej_key)
            rej_out.block_until_ready()
            rejection_meeting_times_res[i, k] = rej_out
            rejection_runtime_res[i, k] = time.time() - tic

    thor_key = JAX_KEY
    for j, C in enumerate(tqdm.tqdm(CS, leave=True)):
        thor_sampler = thorisson_sample(C)
        thor_experiment_fun = jax.jit(lambda op_key: sample_coupled_chain(op_key, EPS, thor_sampler, log_target, dim))

        # run it once to compile
        block = thor_experiment_fun(thor_key)
        block.block_until_ready()
        for k in range(K):
            tic = time.time()
            thor_key, subkey = jax.random.split(thor_key)
            thor_out = thor_experiment_fun(thor_key)
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

    index = pd.MultiIndex.from_product([["meeting time", "run time (s)"], ["mean", "standard deviation"]])
    thorisson_df = pd.DataFrame(columns=data["CS"], index=index)
    rejection_df = pd.DataFrame(columns=data["NS"], index=index)

    thorisson_df.loc[("meeting time", "mean")] = [f"${v.mean(-1):.1f}$" for v in data["thorisson_meeting_times"]]
    thorisson_df.loc[("run time (s)", "mean")] = [f"${v.mean(-1):.1e}$" for v in data["thorisson_runtime"][:, 1:]]

    rejection_df.loc[("meeting time", "mean")] = [f"${v.mean(-1):.1f}$" for v in data["rejection_meeting_times"]]
    rejection_df.loc[("run time (s)", "mean")] = [f"${v.mean(-1):.1e}$" for v in data["rejection_runtime"][:, 1:]]

    thorisson_df.loc[("meeting time", "standard deviation")] = [f"${v.std(-1):.1f}$" for v in
                                                                data["thorisson_meeting_times"]]
    thorisson_df.loc[("run time (s)", "standard deviation")] = [f"${v.std(-1):.1e}$" for v in
                                                                data["thorisson_runtime"][:, 1:]]

    rejection_df.loc[("meeting time", "standard deviation")] = [f"${v.std(-1):.1f}$" for v in
                                                                data["rejection_meeting_times"]]
    rejection_df.loc[("run time (s)", "standard deviation")] = [f"${v.std(-1):.1e}$" for v in
                                                                data["rejection_runtime"][:, 1:]]

    print(rejection_df.to_latex("out/rejection_mmala.tex"))
    print(thorisson_df.to_latex("out/thorisson_mmala.tex"))
