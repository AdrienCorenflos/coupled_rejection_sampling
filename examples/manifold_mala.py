"""
Riemannian manifold MALA logistic regression example. This corresponds to the Application 4 of [1].
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import kalepy as kale
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from jax.scipy.stats import norm
from matplotlib import pyplot as plt

from coupled_rejection_sampling.mvn import coupled_mvns, mvn_logpdf
from coupled_rejection_sampling.reflected_coupled_mh import reflection_coupled_mh
from coupled_rejection_sampling.thorisson import modified_thorisson

JAX_KEY = jax.random.PRNGKey(0)
K = 100_000  # number of experiments
CS = np.linspace(0.8, 1., 3)
NS = [1, 4, 16, 64]

ALPHA = 100.  # Same wide prior as in mMALA paper
EPS = 1.

INIT_STD = 0.25

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
def sample_rejection_coupled_chain(key, eps, sampler, log_pi, D):
    key, init_x_key, init_y_key = jax.random.split(key, 3)

    x0 = INIT_STD * jax.random.normal(init_x_key, (D,))
    y0 = INIT_STD * jax.random.normal(init_y_key, (D,))

    def cond(carry):
        return ~carry[-1]

    def body(carry):
        op_key, x, y, iteration, _ = carry
        op_key, sample_key = jax.random.split(op_key, 2)
        x, y, coupled = simplified_manifold_mala_step(sample_key, x, y, eps, sampler, log_pi)
        return op_key, x, y, iteration + 1, coupled

    *_, meeting_time, _ = jax.lax.while_loop(cond, body, (key, x0, y0, 0, False))
    return meeting_time


def get_proposal_logpdf(log_pi, eps):
    def proposal_logpdf(x, x_prime):
        prop_mean, prop_chol = _get_manifold_langevin_discretisation(x, eps, log_pi)
        return mvn_logpdf(x_prime, prop_mean, prop_chol)

    return proposal_logpdf


def get_proposal_sampler(log_pi, eps):
    def proposal_sampler(k, x):
        prop_mean, prop_chol = _get_manifold_langevin_discretisation(x, eps, log_pi)
        return prop_mean + prop_chol @ jax.random.normal(k, x.shape)

    return proposal_sampler


@partial(jax.jit, static_argnums=(2, 3))
def sample_mh_reflected_chain(key, eps, log_pi, D):
    key, init_x_key, init_y_key = jax.random.split(key, 3)

    proposal_logpdf = get_proposal_logpdf(log_pi, eps)
    proposal_sampler = get_proposal_sampler(log_pi, eps)
    step = reflection_coupled_mh(proposal_sampler, proposal_logpdf, log_pi)

    x0 = INIT_STD * jax.random.normal(init_x_key, (D,))
    y0 = INIT_STD * jax.random.normal(init_y_key, (D,))

    def cond(carry):
        return ~carry[-1]

    def body(carry):
        op_key, x, y, iteration, _ = carry
        op_key, sample_key = jax.random.split(op_key, 2)
        (x, _), (y, _), coupled = step(sample_key, x, y)
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
    reflected_mh_meeting_times_res = np.empty((K,))

    rejection_runtime_res = np.empty((len(NS), K))
    reflected_mh_runtime_res = np.empty((K,))

    rej_key = JAX_KEY
    for i, N in enumerate(NS):
        rej_sampler = rejection_sample(N)
        rej_experiment_fun = jax.jit(
            lambda op_key: sample_rejection_coupled_chain(op_key, EPS, rej_sampler, log_target, dim))

        # run it once to compile
        block = rej_experiment_fun(rej_key)
        block.block_until_ready()
        for k in tqdm.trange(K, desc=f'Rejection sampling N={N}'):
            tic = time.time()
            rej_key, subkey = jax.random.split(rej_key)

            rej_out = rej_experiment_fun(rej_key)
            rej_out.block_until_ready()
            rejection_meeting_times_res[i, k] = rej_out
            rejection_runtime_res[i, k] = time.time() - tic

    reflected_mh_key = JAX_KEY
    reflected_mh_experiment_fun = jax.jit(
        lambda op_key: sample_mh_reflected_chain(op_key, EPS, log_target, dim))
    block = reflected_mh_experiment_fun(reflected_mh_key)
    block.block_until_ready()
    for k in tqdm.trange(K, desc=f'Full kernel sampling'):
        tic = time.time()
        reflected_mh_key, subkey = jax.random.split(reflected_mh_key)
        reflected_mh_out = reflected_mh_experiment_fun(reflected_mh_key)
        reflected_mh_out.block_until_ready()
        reflected_mh_meeting_times_res[k] = reflected_mh_out
        reflected_mh_runtime_res[k] = time.time() - tic

    return rejection_meeting_times_res, rejection_runtime_res, reflected_mh_meeting_times_res, reflected_mh_runtime_res


if RUN:
    rejection_meeting_times, rejection_runtime, reflected_mh_meeting_times, reflected_mh_runtime = experiment()
    np.savez(save_path, CS=CS, NS=NS,
             rejection_meeting_times=rejection_meeting_times, rejection_runtime=rejection_runtime,
             reflected_mh_meeting_times=reflected_mh_meeting_times, reflected_mh_runtime=reflected_mh_runtime)

if PLOT:
    cmap = plt.get_cmap("tab10")
    data = np.load(save_path)

    for i, N in enumerate(NS):
        points, density = kale.density(data["rejection_meeting_times"][i, :], points=None, reflect=[True, None])
        plt.plot(points, density, lw=2.0, alpha=0.8, label=f'ERS N={N}')
    points, density = kale.density(data["reflected_mh_meeting_times"], points=None, reflect=[True, None])
    plt.plot(points, density, lw=2.0, alpha=0.8, label=r"\citet{}")
    plt.xlim(0, 80)
    plt.legend()
    plt.show()

    for i, N in enumerate(NS):
        points, density = kale.density(data["rejection_runtime"][i, 1:], points=None, reflect=[True, None])
        plt.plot(points, density, lw=2.0, alpha=0.8, label=f'ERS N={N}')
    points, density = kale.density(data["reflected_mh_runtime"][1:], points=None, reflect=[True, None])
    plt.plot(points, density, lw=2.0, alpha=0.8, label=r"\citet{}")
    plt.xlim(0, 0.08)
    plt.legend()
    plt.show()

    index = ["meeting time", "run time (s)"]
    rejection_df_index = pd.MultiIndex.from_product([index, NS])
    rejection_df = pd.DataFrame(columns=np.arange(K, dtype=int), index=rejection_df_index, dtype=float)
    reflected_mh_index = pd.MultiIndex.from_product([index, [""]])
    reflected_mh_df = pd.DataFrame(columns=np.arange(K, dtype=int), index=reflected_mh_index, dtype=float)

    for i, N in enumerate(NS):
        rejection_df.loc[("meeting time", N)] = data["rejection_meeting_times"][i]
        rejection_df.loc[("run time (s)", N)] = data["rejection_runtime"][i]

    reflected_mh_df.loc[("meeting time", "")] = data["reflected_mh_meeting_times"]
    reflected_mh_df.loc[("run time (s)", "")] = data["reflected_mh_runtime"]

    reflected_mh_df = reflected_mh_df.T.describe(percentiles=[0.05, 0.95]).T
    rejection_df = rejection_df.T.describe(percentiles=[0.05, 0.95]).T

    reflected_mh_df = reflected_mh_df.drop(["count", "min", "max"], axis=1)
    rejection_df = rejection_df.drop(["count", "min", "max"], axis=1)

    df = pd.concat([rejection_df, reflected_mh_df]).sort_index()
    print(df.loc["meeting time"].to_latex("out/meeting_time_mmala.tex", float_format='%.1f'))
    print(df.loc["run time (s)"].to_latex("out/runtime_mmala.tex", float_format='%.1e'))


    # index = pd.MultiIndex.from_product([["meeting time", "run time (s)"], ["mean", "standard deviation"]])
    # rejection_df = pd.DataFrame(columns=data["NS"], index=index)
    # reflected_mh_df = pd.Series(index=index, dtype=float)
    #
    #
    # rejection_df.loc[("meeting time", "mean")] = [f"${v.mean(-1):.1f}$" for v in data["rejection_meeting_times"]]
    # rejection_df.loc[("run time (s)", "mean")] = [f"${v.mean(-1):.1e}$" for v in data["rejection_runtime"][:, 1:]]
    #
    # reflected_mh_df.loc[("meeting time", "mean")] = f"${data['reflected_mh_meeting_times'].mean(-1):.1f}$"
    # reflected_mh_df.loc[("run time (s)", "mean")] = f"${data['reflected_mh_runtime'].mean(-1):.1e}$"
    #
    # rejection_df.loc[("meeting time", "standard deviation")] = [f"${v.std(-1):.1f}$" for v in
    #                                                             data["rejection_meeting_times"]]
    # rejection_df.loc[("run time (s)", "standard deviation")] = [f"${v.std(-1):.1e}$" for v in
    #                                                             data["rejection_runtime"][:, 1:]]
    #
    # reflected_mh_df.loc[("meeting time", "standard deviation")] = f"${data['reflected_mh_meeting_times'].std(-1):.1f}$"
    # reflected_mh_df.loc[("run time (s)", "standard deviation")] = f"${data['reflected_mh_runtime'].std(-1):.1e}$"
    #
    # print(rejection_df.to_latex("out/rejection_mmala.tex"))
    # print(reflected_mh_df.to_latex("out/reflected_mh_mmala.tex"))
