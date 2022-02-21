import jax.numpy as jnp
import jax.random
import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as np_test
import pytest
from scipy.integrate import quad

from coupled_rejection_sampling.reflected_coupled_mh import _transport_fn, _get_log_f, reflection_coupled_mh


def test_transport_fn():
    np.random.seed(0)
    x = np.random.randn(2)
    y = np.random.randn(2)
    e = (y - x) / np.linalg.norm(y - x)

    vec_transport_fn = jnp.vectorize(_transport_fn, signature="(d),(d),(d)->(d)")
    thetas = jnp.linspace(0, 2 * np.pi, 10)

    zs = np.stack([np.cos(thetas), np.sin(thetas)], 1)

    out = vec_transport_fn(x, y, zs)
    np_test.assert_allclose(np.dot(out - x[None, :], e), -np.dot(zs - x, e), atol=1e-6)


def test_log_f():
    # This test should be an assertion on (e.g.) a KS test, but I'm feeling lazy...
    np.random.seed(0)
    N = 100_000
    alpha = 1.2
    log_pi = jstats.laplace.logpdf
    log_proposal = lambda x, x_prime: jstats.norm.logpdf(x_prime - alpha * x)

    x_init = np.random.randn()

    log_f = _get_log_f(log_proposal, log_pi)
    Z = quad(lambda x_prime: np.exp(log_f(x_init, x_prime)), -np.inf, np.inf)[0]

    proposals = np.random.randn(N) + alpha * x_init
    log_accept_ratio = log_pi(proposals) + log_proposal(proposals, x_init) - log_pi(x_init) - log_proposal(x_init,
                                                                                                           proposals)

    log_us = np.log(np.random.rand(N))
    proposals_accepted = proposals[log_us < log_accept_ratio]

    plt.hist(proposals_accepted, bins=100, density=True, alpha=0.25)
    sorted_proposals_accepted = np.sort(proposals_accepted)

    plt.plot(sorted_proposals_accepted, np.exp(log_f(x_init, sorted_proposals_accepted)) / Z)
    plt.show()


def test_reflection_coupled_mh_full():
    np.random.seed(0)
    N = 10_000
    alpha = 1.2
    log_pi = jstats.laplace.logpdf
    log_proposal = lambda x, x_prime: jstats.norm.logpdf(x_prime - alpha * x)
    proposal = lambda k, x: alpha * x + jax.random.normal(k)

    step = reflection_coupled_mh(proposal, log_proposal, log_pi)

    x_init = 2 * np.random.randn(N)
    y_init = 3 * np.random.randn(N)
    keys = jax.random.split(jax.random.PRNGKey(42), N)

    @jax.vmap
    def sample_rejection_coupled_chain(key, x0, y0):

        def cond(carry):
            *_, coupled, _ = carry
            return ~coupled

        def body(carry):
            op_key, x, y, _, i = carry
            op_key, sample_key = jax.random.split(op_key, 2)
            (x, _), (y, _), coupled = step(sample_key, x, y)
            return op_key, x, y, coupled, i+1

        *_, n_iter = jax.lax.while_loop(cond, body, (key, x0, y0, False, 0))
        return n_iter

    coupling_times = sample_rejection_coupled_chain(keys, x_init, y_init)
    coupling_times = np.asarray(coupling_times)

    print(np.mean(coupling_times))
    print(np.std(coupling_times))

    plt.hist(coupling_times, bins=100)
    plt.xscale("log")
    plt.show()

@pytest.mark.parametrize("seed", [0])
def test_reflection_coupled_mh_one_step(seed):
    # This test should be an assertion on (e.g.) a KS test, but I'm feeling lazy...

    np.random.seed(seed)
    N = 1_000_000
    alpha = 1.2
    log_pi = jstats.laplace.logpdf
    log_proposal = lambda x, x_prime: jstats.norm.logpdf(x_prime - alpha * x)
    proposal = lambda k, x: alpha * x + jax.random.normal(k)

    step = reflection_coupled_mh(proposal, log_proposal, log_pi)

    x_init = np.random.randn()
    y_init = np.random.randn()
    keys = jax.random.split(jax.random.PRNGKey(42), N)

    (xs, accepted_x), (ys, accepted_y), coupled = jax.vmap(step, in_axes=[0, None, None])(keys, x_init, y_init)

    print(np.asarray(ys[accepted_y]))
    print(np.asarray(ys[~accepted_y]))

    xs = np.asarray(xs[accepted_x])
    ys = np.asarray(ys[accepted_y])

    log_f = _get_log_f(log_proposal, log_pi)
    Z = quad(lambda y_prime: np.exp(log_f(y_init, y_prime)), -np.inf, np.inf)[0]

    plt.hist(ys, bins=100, density=True, alpha=0.25)
    sorted_proposals_accepted = np.sort(ys)
    plt.plot(sorted_proposals_accepted, np.exp(log_f(y_init, sorted_proposals_accepted)) / Z)
    plt.show()
    plt.title("Y")
    #
    Z = quad(lambda x_prime: np.exp(log_f(x_init, x_prime)), -np.inf, np.inf)[0]

    plt.hist(xs, bins=100, density=True, alpha=0.25)
    sorted_proposals_accepted = np.sort(xs)
    plt.plot(sorted_proposals_accepted, np.exp(log_f(x_init, sorted_proposals_accepted)) / Z)
    plt.title("X")
    plt.show()
