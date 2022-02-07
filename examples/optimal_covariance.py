"""
This corresponds to the Application 1 of [1]. 
"""
import time
from functools import partial

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm
from jax.scipy import linalg

from coupled_rejection_sampling.mvn import get_optimal_covariance, coupled_mvns, lower_bound, tril_log_det

RUN = False
PLOT = True
save_path = "out/optimal.npz"

jax_key = jax.random.PRNGKey(42)
D = 15
P = jnp.diag(jnp.arange(1, D + 1))

chol_P = jnp.linalg.cholesky(P)
m = mu = jnp.zeros((D,))

B = 200  # number of covariances
M = 500  # number of RS experiments
NS = [4 ** k for k in range(6)]


def rejection_experiment():
    batch_keys = jax.random.split(jax_key, B)

    @jax.jit
    def get_chol_covs(k):
        gauss = jax.random.normal(k, (D, D))
        orth, _ = linalg.qr(gauss)
        Sigma = orth @ P @ orth.T
        chol_Sigma = jnp.linalg.cholesky(Sigma)
        chol_Q = get_optimal_covariance(chol_P, chol_Sigma)
        return chol_Sigma, chol_Q

    res_shape = B, len(NS), 2
    res_mean_coupled = np.zeros(res_shape)
    res_var_coupled = np.zeros(res_shape)
    res_mean_trials = np.zeros(res_shape)
    res_var_trials = np.zeros(res_shape)
    res_runtime = np.zeros(res_shape)

    res_coupling_bounds = np.zeros((B, 2))
    res_trials_bounds = np.zeros(res_shape)

    @partial(jax.jit, static_argnums=(3,))
    def test_fun(op_key, chol_Sigma, chol_Q, N):
        keys = jax.random.split(op_key, M)
        vmapped_coupled_mvns = jax.jit(
            jax.vmap(lambda k: coupled_mvns(k, m, chol_P, mu, chol_Sigma, N, chol_Q)))
        *_, are_coupled, n_trials = vmapped_coupled_mvns(keys)
        return np.mean(are_coupled), np.var(are_coupled), np.mean(n_trials), np.var(n_trials)

    for i, n in enumerate(NS):
        for b in tqdm.trange(B):
            mat_key, rs_key = jax.random.split(batch_keys[b])
            chol_Sigma, chol_Q = get_chol_covs(mat_key)

            chol_Q_max = jnp.max(jnp.maximum(jnp.diag(chol_P), jnp.diag(chol_Sigma))) * jnp.eye(D)

            trials_opt = np.exp(np.minimum(tril_log_det(chol_Q) - tril_log_det(chol_Sigma),
                                           tril_log_det(chol_Q) - tril_log_det(chol_P)))
            trials_max = np.exp(np.minimum(tril_log_det(chol_Q_max) - tril_log_det(chol_Sigma),
                                           tril_log_det(chol_Q_max) - tril_log_det(chol_P)))

            if n == 1:
                res_coupling_bounds[b, 0] = lower_bound(m, chol_P, mu, chol_Sigma, chol_Q)
                res_coupling_bounds[b, 1] = lower_bound(m, chol_P, mu, chol_Sigma, chol_Q_max)

            res_trials_bounds[b, i, 0] = (1 + (n - 1) / trials_opt) / (n / trials_opt)
            res_trials_bounds[b, i, 1] = (1 + (n - 1) / trials_max) / (n / trials_max)

            tic = time.time()
            (res_mean_coupled[b, i, 0], res_var_coupled[b, i, 0], res_mean_trials[b, i, 0],
             res_var_trials[b, i, 0]) = test_fun(rs_key, chol_Sigma, chol_Q, n)
            res_runtime[b, i, 0] = time.time() - tic

            tic = time.time()
            (res_mean_coupled[b, i, 1], res_var_coupled[b, i, 1], res_mean_trials[b, i, 1],
             res_var_trials[b, i, 1]) = test_fun(rs_key, chol_Sigma, chol_Q_max, n)
            res_runtime[b, i, 1] = time.time() - tic
    return res_mean_coupled, res_var_coupled, res_mean_trials, res_var_trials, res_coupling_bounds, res_trials_bounds, res_runtime


if RUN:
    (coupled_mean, coupled_var, n_trials_mean, n_trials_var, coupling_bounds, n_trials_bounds,
     runtime) = rejection_experiment()
    np.savez(save_path, coupled_mean=coupled_mean, coupled_var=coupled_var,
             n_trials_mean=n_trials_mean, n_trials_var=n_trials_var, coupling_bounds=coupling_bounds,
             n_trials_bounds=n_trials_bounds, runtime=runtime)

if PLOT:
    data = np.load(save_path)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7), sharex=True, sharey=True)

    print(np.mean(data["runtime"][1:], 0))
    print(np.std(data["runtime"][1:], 0))

    for i, n in enumerate(NS):
        ax = axes.flatten()[i]
        ax.set_title(f"$N={n}$")
        arg_sort = np.argsort(data["coupling_bounds"][:, 0])
        ax.scatter(range(B), data["coupled_mean"][arg_sort, i, 0], label=f"Empirical Optimised",
                   color="tab:blue", alpha=0.75
                   )
        ax.scatter(range(B), data["coupled_mean"][arg_sort, i, 1], label=f"Empirical MAX",
                   color="tab:orange",
                   alpha=0.75)
        twinx = ax.twinx()
        twinx.semilogy(range(B), data["coupling_bounds"][arg_sort, 0], label="Optimised bound",
                   color="tab:blue")
        twinx.semilogy(range(B), data["coupling_bounds"][arg_sort, 1], label="MAX bound",
                   color="tab:orange")
    axes[0, 0].legend(loc="upper left")
    plt.show()
    # tikzplotlib.save("out/gaussian_opt_coupling.tikz")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7), sharex=True, sharey=True)
    for i, n in enumerate(NS):
        ax = axes.flatten()[i]
        ax.set_title(f"$N={n}$")
        arg_sort = np.argsort(data["n_trials_bounds"][:, i, 0])[::-1]
        ax.scatter(range(B), 1 / data["n_trials_mean"][arg_sort, i, 0], label="Empirical Optimised",
                   color="tab:blue", alpha=0.75)
        ax.plot(range(B), 1 / data["n_trials_bounds"][arg_sort, i, 0], label="Optimised bound",
                color="tab:blue")

        ax.scatter(range(B), 1 / data["n_trials_mean"][arg_sort, i, 1], label="Empirical MAX", color="tab:orange",
                   alpha=0.75)
        ax.plot(range(B), 1 / data["n_trials_bounds"][arg_sort, i, 1], label="MAX bound", color="tab:orange")
    axes[0, 0].legend(loc="upper left")
    plt.show()

    # tikzplotlib.save("out/gaussian_opt_acceptance.tikz")
