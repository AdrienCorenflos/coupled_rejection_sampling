# Coupled sampling from 2 different tails x > mu and x > eta of N(x | 0,1) Gaussian
# by using a maximal coupling of translated exponential proposals hatp(x) = alpha exp(-alpha (x - mu))
# and hatq(x) = beta exp(-beta (x - eta)) similarly to Robert (1995), but with the coupling.
#
# Robert, C. P. (1995). Simulation of truncated normal variables. Statistics and Computing, Volume 5, pages 121–125.

import math
from functools import partial

import jax.numpy as jnp
import jax.random
import jax.lax
import jax.scipy.stats as jstats

from coupled_rejection_sampling.coupled_rejection_sampler import coupled_sampler


def get_alpha(mu):
    """ Compute the optimal alpha as per Robert (1995) """
    return 0.5 * (mu + jnp.sqrt(mu ** 2 + 4))

def get_gamma(mu, eta, alpha, beta):
    """ Threshold when hatp(x) = hatq(x) """
    return (jnp.log(beta) - jnp.log(alpha) + beta * eta - alpha * mu) / (beta - alpha)

def texp_logpdf(x, mu, alpha):
    """ Translated exponential density """
    return jnp.where(x < mu, -jnp.inf, jnp.log(alpha) - alpha * (x - mu))


def gamma_logpdf(x, a, b, mu):
    """ Gamma density """
    return jnp.where(x >= mu,
                     a * jnp.log(b) - jax.scipy.special.gammaln(a) + (a - 1.0) * jnp.log(x - mu) - b * (x - mu),
                     -jnp.inf)

def gamma_random(key, a, b, mu, N=1):
    """ Draw Gamma random variables """
    return mu + jax.random.gamma(key, a, shape=(N,)) / b


class GaussTails:

    def __init__(self, mu, eta):
        if mu >= eta:
            # We could in principle just swap, but then it swaps tp(x) and tq(x) as well. Thus
            # let the upper level worry about this.
            raise ValueError("Must have eta > mu")

        self.alpha = get_alpha(mu)
        self.beta = get_alpha(eta)
        self.gamma = get_gamma(mu, eta, self.alpha, self.beta)

        self.eta = eta
        self.mu = mu

        self.e_alpha_eta_mu = jnp.exp(-self.alpha * (self.eta - self.mu))
        self.e_alpha_gamma_mu = jnp.exp(-self.alpha * (self.gamma - self.mu))
        self.e_beta_gamma_eta = jnp.exp(-self.beta * (self.gamma - self.eta))

    def pxy(self):
        """" Maximal coupling probability P(X=Y) for translated exponentials """
        return self.e_alpha_eta_mu - self.e_alpha_gamma_mu + self.e_beta_gamma_eta

    def c_logpdf(self, x):
        """ Evaluate c(x) \propto min(hatp(x), hatq(x)) """
        Z = self.pxy()
        return jnp.where((x >= self.eta) & (x <= self.gamma),
                         texp_logpdf(x, self.mu, self.alpha) - jnp.log(Z),
                         texp_logpdf(x, self.eta, self.beta) - jnp.log(Z))

    def c_sample(self, key, N):
        """ Sample from c(x) \propto min(hatp(x), hatq(x)) """

        def C1_inv(u):
            return self.mu - jnp.log(self.e_alpha_eta_mu
                                     - u * (self.e_alpha_eta_mu - self.e_alpha_gamma_mu)) / self.alpha

        def C2_inv(u):
            return self.gamma - jnp.log(1.0 - u) / self.beta

        p1 = self.e_alpha_eta_mu - self.e_alpha_gamma_mu
        p2 = self.e_beta_gamma_eta
        p = p1 / (p1 + p2)

        u12 = jax.random.uniform(key, shape=(N,2))
        u1 = u12[:,0]
        u2 = u12[:,1]

        return jnp.where(u1 < p, C1_inv(u2), C2_inv(u2))

    def TP1_inv(self, u, max_iter=50, err_thr=1e-10):
        """ Solve the equation (exp(-alpha (x - mu)) - exp(-beta (x - eta))) / tZ = u for x >= gamma
            If u has multiple elements, then the equation is solved for each of them """

        # TODO: This could be reimplemented as vectorized map over u

        tZ = self.e_alpha_gamma_mu - self.e_beta_gamma_eta

        # Initial guess
        q = (self.alpha**2 * self.e_alpha_gamma_mu - self.beta**2 * self.e_beta_gamma_eta) / tZ
        xp1 = self.gamma + jnp.sqrt(2.0 * (u - 1.0) / q)  # Using quadratic fit at x = gamma
        d = (jnp.exp(self.alpha * self.mu) - jnp.exp(self.beta * self.eta) * jnp.exp(-(self.beta - self.alpha) * self.gamma)) / tZ
        xp2 = -(1.0 / self.alpha) * jnp.log(u / d)       # Using lower bounding exponential for x >= gamma
        init_xp = jnp.max(jnp.stack([xp1, xp2]), axis=0)

        def log_f(x):
            return jnp.log((jnp.exp(-self.alpha * (x - self.mu)) - jnp.exp(-self.beta * (x - self.eta))) / tZ) - jnp.log(u)

        def log_df(x):
            return (-self.alpha * jnp.exp(-self.alpha * (x - self.mu)) + self.beta * jnp.exp(-self.beta * (x - self.eta))) \
                   / (jnp.exp(-self.alpha * (x - self.mu)) - jnp.exp(-self.beta * (x - self.eta)))

        def body(carry):
            xp, i, err = carry
            lf = log_f(xp)
            ldf = log_df(xp)
            xp = xp - lf / ldf
            err = jnp.exp(lf + jnp.log(u)) - u  # Actual error
            return xp, i+1, err

        def cond(carry):
            xp, i, err = carry
            return jnp.logical_and(i < max_iter, jnp.abs(err).max() > err_thr)

        return jax.lax.while_loop(cond, body, (init_xp, 0, 100.0 * jnp.ones_like(init_xp)))


    def TQ_inv(self, u, max_iter=50, err_thr=1e-10):
        """ Solve the equation (exp(-beta (x - eta) + exp(-alpha (eta - mu)) - exp(-alpha (x - mu))) / Zq = 1/Zq - u
            for eta <= x <= gamma. If y has multiple elements, then the equation is solved for each of them """

        # TODO: This could be reimplemented as vectorized map over u

        Zq = 1 - self.e_alpha_eta_mu + self.e_alpha_gamma_mu - self.e_beta_gamma_eta

        # Initial guess
        init_xp = self.eta * jnp.ones_like(u)

        def log_f(x):
            return jnp.log((self.e_alpha_eta_mu - jnp.exp(-self.alpha*(x - self.mu))
                            + jnp.exp(-self.beta * (x - self.eta))) / Zq) - jnp.log(1.0/Zq-u)

        def log_df(x):
            return (self.alpha * jnp.exp(-self.alpha*(x - self.mu)) - self.beta * jnp.exp(-self.beta * (x - self.eta))) \
                   / (self.e_alpha_eta_mu - jnp.exp(-self.alpha * (x-self.mu)) + jnp.exp(-self.beta * (x - self.eta)))

        def body(carry):
            xp, i, err = carry
            lf = log_f(xp)
            ldf = log_df(xp)
            xp = xp - lf / ldf
            err = jnp.exp(lf + jnp.log(1.0/Zq-u)) - 1.0/Zq + u  # Actual error
            return xp, i+1, err

        def cond(carry):
            xp, i, err = carry
            return jnp.logical_and(i < max_iter, jnp.abs(err).max() > err_thr)

        return jax.lax.while_loop(cond, body, (init_xp, 0, 100.0 * jnp.ones_like(init_xp)))


    def tp_logpdf(self, x):
        """ Evaluate tildep(x) \propto p(x) - min(p(x),q(x)) """
        Zp = 1 - self.e_alpha_eta_mu + self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        # XXX: Note that this evaluates some logs of negative values which might cause trouble:
        return jnp.where((x >= self.mu) & (x <= self.eta), texp_logpdf(x, self.mu, self.alpha) - jnp.log(Zp),
                         jnp.where(x >= self.gamma,
                                   jnp.log(jnp.exp(texp_logpdf(x, self.mu, self.alpha))
                                           - jnp.exp(texp_logpdf(x, self.eta, self.beta))) - jnp.log(Zp), -jnp.inf))

    def tp_sample_icdf(self, key, N=1):
        """ Sample from tildep(x) \propto p(x) - min(p(x),q(x)) using icdf method """
        u12 = jax.random.uniform(key, shape=(N,2))
        u1 = u12[:,0]
        u2 = u12[:,1]

        p1 = self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        p2 = 1 - self.e_alpha_eta_mu
        p = p1 / (p1 + p2)

        def TP2_inv(u):
            return self.mu - (1.0/self.alpha) * jnp.log(1.0 - u * (1 - self.e_alpha_eta_mu))

        return jnp.where(u1 < p, self.TP1_inv(u2)[0], TP2_inv(u2))

    def tp1_sample_rs(self, key, N=1):
        """ Sample from tildep1(x) with rejection sampling method """
        tZ = self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        M = (self.beta / tZ) * self.e_beta_gamma_eta * (self.beta - self.alpha) / self.alpha ** 2

        def tp1_sample_rs_one(k):
            def body(carry):
                curr_k, *_ = carry
                curr_k, k2, k3 = jax.random.split(curr_k, 3)
                x = gamma_random(k2, 2, self.alpha, self.gamma)[0]
                dens1 = jnp.where(x >= self.gamma,
                                  (jnp.exp(texp_logpdf(x, self.mu, self.alpha))
                                   - jnp.exp(texp_logpdf(x, self.eta, self.beta))) / tZ, 0.0)
                dens2 = jnp.exp(gamma_logpdf(x, 2, self.alpha, self.gamma))
                u = jax.random.uniform(k3)
                accepted = u <= (dens1 / dens2 / M)
                return curr_k, x, accepted

            _, x_out, _ = jax.lax.while_loop(lambda carry: jnp.logical_not(carry[-1]), body, (k, 0.0, False))
            return x_out

        keys = jax.random.split(key, N)
        x_values = jax.vmap(tp1_sample_rs_one)(keys)
        return x_values


    def tp_sample_rs(self, key, N=1):
        """ Sample from tildep(x) \propto p(x) - min(p(x),q(x)) using RS method """
        key1, key2 = jax.random.split(key, 2)
        u12 = jax.random.uniform(key1, shape=(N,2))
        u1 = u12[:,0]
        u2 = u12[:,1]

        p1 = self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        p2 = 1 - self.e_alpha_eta_mu
        p = p1 / (p1 + p2)

        def TP2_inv(u):
            return self.mu - (1.0/self.alpha) * jnp.log(1.0 - u * (1 - self.e_alpha_eta_mu))

        return jnp.where(u1 < p, self.tp1_sample_rs(key2, N), TP2_inv(u2))


    def tq_logpdf(self, x):
        """ Evaluate tildeq(x) \propto q(x) - min(p(x),q(x)) """
        Zq = 1 - self.e_alpha_eta_mu + self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        # XXX: Note that this evaluates some logs of negative values which might cause trouble:
        return jnp.where((x >= self.eta) & (x <= self.gamma),
                         jnp.log(jnp.exp(texp_logpdf(x, self.eta, self.beta))
                                 - jnp.exp(texp_logpdf(x, self.mu, self.alpha))) - jnp.log(Zq), -jnp.inf)

    def tq_sample_icdf(self, key, N=1):
        """ Sample from tildeq(x) \propto q(x) - min(p(x),q(x)) with icdf method """
        u = jax.random.uniform(key, shape=(N,))
        return self.TQ_inv(u)[0]


    def tq_sample_rs(self, key, N=1):
        """ Sample from tildeq(x) \propto q(x) - min(p(x),q(x)) with rejection sampling method """
        Zq = 1 - self.e_alpha_eta_mu + self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        M = 1 / Zq

        def tq_sample_rs_one(k):
            def body(carry):
                curr_k, *_ = carry
                curr_k, k2, k3 = jax.random.split(curr_k, 3)
                u1 = jax.random.uniform(k2)
                x = self.eta - (1.0 / self.beta) * jnp.log(1 - u1)
                u2 = jax.random.uniform(k3)
                dens1 = jnp.where((x >= self.eta) & (x <= self.gamma),
                                  (jnp.exp(texp_logpdf(x, self.eta, self.beta))
                                   - jnp.exp(texp_logpdf(x, self.mu, self.alpha))) / Zq, 0.0)
                dens2 = jnp.exp(texp_logpdf(x, self.eta, self.beta))
                accepted = u2 <= dens1 / dens2 / M
                return curr_k, x, accepted

            _, x_out, _ = jax.lax.while_loop(lambda carry: jnp.logical_not(carry[-1]), body, (k, 0.0, False))
            return x_out

        keys = jax.random.split(key, N)
        x_values = jax.vmap(tq_sample_rs_one)(keys)
        return x_values


    def Gamma_hat_icdf(self, key, N=1):
        """ Sample from coupling of translated exponentials with icdf methods """
        # TODO: no unit test for this yet
        pxy = self.pxy()
        new_keys = jax.random.split(key, 4)

        u = jax.random.uniform(new_keys[0], shape=(N,))
        are_coupled = (u <= pxy)

        c_samples = self.c_sample(new_keys[1], N)
        tp_samples = self.tp_sample_icdf(new_keys[2], N)
        tq_samples = self.tq_sample_icdf(new_keys[3], N)

        xs = jnp.where(are_coupled, c_samples, tp_samples)
        ys = jnp.where(are_coupled, c_samples, tq_samples)
        return xs, ys, are_coupled



    def Gamma_hat_rs(self, key, N=1):
        """ Sample from coupling of translated exponentials with part RS methods """
        # TODO: no unit test for this yet
        pxy = self.pxy()
        new_keys = jax.random.split(key, 4)

        u = jax.random.uniform(new_keys[0], shape=(N,))
        are_coupled = (u <= pxy)

        c_samples = self.c_sample(new_keys[1], N)
        tp_samples = self.tp_sample_rs(new_keys[2], N)
        tq_samples = self.tq_sample_rs(new_keys[3], N)

        xs = jnp.where(are_coupled, c_samples, tp_samples)
        ys = jnp.where(are_coupled, c_samples, tq_samples)
        return xs, ys, are_coupled

    #
    # Samplers for marginals by using the method from Robert (1995)
    #
    def p(self, key, N=1):
        # TODO: Get rid of the vmap
        # TODO: No unit test yet
        def p_one(k):
            def body(carry):
                curr_k, *_ = carry
                curr_k, k2, k3 = jax.random.split(curr_k, 3)
                u1 = jax.random.uniform(k2)
                x = self.mu - (1.0 / self.alpha) * jnp.log(1 - u1)
                u2 = jax.random.uniform(k3)
                accepted = u2 <= jnp.exp(-0.5 * (x - self.alpha) ** 2)
                return curr_k, x, accepted

            _, x_out, _ = jax.lax.while_loop(lambda carry: jnp.logical_not(carry[-1]), body, (k, 0.0, False))
            return x_out

        keys = jax.random.split(key, N)
        x_values = jax.vmap(p_one)(keys)
        return x_values

    def q(self, key, N=1):
        # TODO: Get rid of the vmap
        # TODO: No unit test yet
        def q_one(k):
            def body(carry):
                curr_k, *_ = carry
                curr_k, k2, k3 = jax.random.split(curr_k, 3)
                u1 = jax.random.uniform(k2)
                x = self.eta - (1.0 / self.beta) * jnp.log(1 - u1)
                u2 = jax.random.uniform(k3)
                accepted = u2 <= jnp.exp(-0.5 * (x - self.beta) ** 2)
                return curr_k, x, accepted

            _, x_out, _ = jax.lax.while_loop(lambda carry: jnp.logical_not(carry[-1]), body, (k, 0.0, False))
            return x_out

        keys = jax.random.split(key, N)
        x_values = jax.vmap(q_one)(keys)
        return x_values

    #
    # ATTN: We actually put log_p = log( p / p_hat ), log_p_hat = 0, M_p = 1 and the same for q
    # to avoid explicit evaluation of p and p_hat and M.
    #
    def log_p(self, x):
        return -0.5 * (x - self.alpha) ** 2

    def log_q(self, x):
        return -0.5 * (x - self.beta) ** 2

    def log_p_hat(self, x):
        return jnp.zeros_like(x)

    def log_q_hat(self, x):
        return jnp.zeros_like(x)

    #
    # The actual sampling routines
    #
    def coupled_gauss_tails_icdf(self, key, N=1):
        # TODO: Not tested at all
        return coupled_sampler(key, self.Gamma_hat_icdf, lambda x: self.p(x)[0], lambda x: self.q(x)[0], self.log_p_hat, self.log_q_hat,
                               self.log_p, self.log_q, 0.0, 0.0, N)

    def coupled_gauss_tails_rs(self, key, N=1):
        # TODO: Not tested at all
        return coupled_sampler(key, self.Gamma_hat_rs, lambda x: self.p(x)[0], lambda x: self.q(x)[0], self.log_p_hat, self.log_q_hat,
                               self.log_p, self.log_q, 0.0, 0.0, N)

