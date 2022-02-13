# Coupled sampling from 2 different tails x > mu and x > eta of N(x | 0,1) Gaussian
# by using a maximal coupling of translated exponential proposals hatp(x) = alpha exp(-alpha (x - mu))
# and hatq(x) = beta exp(-beta (x - eta)) similarly to Robert (2009), but with the coupling.

import math
from functools import partial

import jax.numpy as jnp
import jax.random
import jax.lax

from coupled_rejection_sampling.coupled_rejection_sampler import coupled_sampler


def get_alpha(mu):
    return 0.5 * (mu + jnp.sqrt(mu ** 2 + 4))

def get_gamma(mu, eta, alpha, beta):
    return (jnp.log(beta) - jnp.log(alpha) + beta * eta - alpha * mu) / (beta - alpha)

def texp_logpdf(x, mu, alpha):
    return jnp.where(x < mu, -jnp.inf, jnp.log(alpha) - alpha * (x - mu))

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
        """" Maximal coupling probability P(X=Y) """
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

    def TP1_inv(self, u, max_iter=20, err_thr=1e-10):
        """ Solve the equation (exp(-alpha (x - mu)) - exp(-beta (x - eta))) / tZ = u for x >= gamma
            If u has multiple elements, then the equation is solved for each of them """

        tZ = self.e_alpha_gamma_mu - self.e_beta_gamma_eta

        # Sanity checks
        if jnp.any(u >= 1):
            raise ValueError("Need to have u < 1")
        if jnp.any(u <= 0):
            raise ValueError("Need to have u > 0")

        # Initial guess
        q = (self.alpha**2 * self.e_alpha_gamma_mu - self.beta**2 * self.e_beta_gamma_eta) / tZ
        xp1 = self.gamma + jnp.sqrt(2.0 * (u - 1.0) / q)  # Using quadratic fit at x = gamma
        d = (jnp.exp(self.alpha * self.mu) - jnp.exp(self.beta * self.eta) * jnp.exp(-(self.beta - self.alpha) * self.gamma)) / tZ
        xp2 = -(1.0 / self.alpha) * jnp.log(u / d)       # Using lower bounding exponential for x >= gamma
        xp = jnp.max(jnp.stack([xp1, xp2]), axis=0)

#        print(f"xp1 = {xp1}, xp2 = {xp2}, xp = {xp}")

        def log_f(x):
            return jnp.log((jnp.exp(-self.alpha * (x - self.mu)) - jnp.exp(-self.beta * (x - self.eta))) / tZ) - jnp.log(u)

        def log_df(x):
            return (-self.alpha * jnp.exp(-self.alpha * (x - self.mu)) + self.beta * jnp.exp(-self.beta * (x - self.eta))) \
                   / (jnp.exp(-self.alpha * (x - self.mu)) - jnp.exp(-self.beta * (x - self.eta)))

        # Newton's iteration
        iter = 0
        err = 100.0 * jnp.ones_like(xp)
        while iter < max_iter and jnp.abs(err).max() > err_thr:
            lf = log_f(xp)
            ldf = log_df(xp)
            xp = xp - lf / ldf
            err = jnp.exp(lf + jnp.log(u)) - u  # Actual error
            iter += 1

#            print(f"xp = {xp}, err = {err}")

        return xp, iter, err

    def TQ_inv(self, u, max_iter=20, err_thr=1e-10):
        """ Solve the equation (exp(-beta (x - eta) + exp(-alpha (eta - mu)) - exp(-alpha (x - mu))) / Zq = 1/Zq - u
            for eta <= x <= gamma. If y has multiple elements, then the equation is solved for each of them """

        Zq = 1 - self.e_alpha_eta_mu + self.e_alpha_gamma_mu - self.e_beta_gamma_eta

        # Sanity checks
        if jnp.any(u >= 1):
            raise ValueError("Need to have u < 1")
        if jnp.any(u <= 0):
            raise ValueError("Need to have u > 0")

        # Initial guess
        xp = self.eta

        def log_f(x):
            return jnp.log((self.e_alpha_eta_mu - jnp.exp(-self.alpha*(x - self.mu))
                            + jnp.exp(-self.beta * (x - self.eta))) / Zq) - jnp.log(1.0/Zq-u)

        def log_df(x):
            return (self.alpha * jnp.exp(-self.alpha*(x - self.mu)) - self.beta * jnp.exp(-self.beta * (x - self.eta))) \
                   / (self.e_alpha_eta_mu - jnp.exp(-self.alpha * (x-self.mu)) + jnp.exp(-self.beta * (x - self.eta)))

        # Newton's iteration
        iter = 0
        err = 100.0 * jnp.ones_like(xp)
        while iter < max_iter and jnp.abs(err).max() > err_thr:
            lf = log_f(xp)
            ldf = log_df(xp)
            xp = xp - lf / ldf
            err = jnp.exp(lf + jnp.log(1.0/Zq-u)) - 1.0/Zq + u  # Actual error
            iter += 1

#            print(f"xp = {xp}, err = {err}")

        return xp, iter, err

    def tp_logpdf(self, x):
        """ Evaluate tildep(x) \propto p(x) - min(p(x),q(x)) """
        Zp = 1 - self.e_alpha_eta_mu + self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        # XXX: Note that this evaluates some logs of negative values which might cause trouble:
        return jnp.where((x >= self.mu) & (x <= self.eta), texp_logpdf(x, self.mu, self.alpha) - jnp.log(Zp),
                         jnp.where(x >= self.gamma,
                                   jnp.log(jnp.exp(texp_logpdf(x, self.mu, self.alpha))
                                           - jnp.exp(texp_logpdf(x, self.eta, self.beta))) - jnp.log(Zp), -jnp.inf))

    def tp_sample_icdf(self, key, N=1):
        """ Sample from tildep(x) \propto p(x) - min(p(x),q(x)) """
        u12 = jax.random.uniform(key, shape=(N,2))
        u1 = u12[:,0]
        u2 = u12[:,1]

        p1 = self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        p2 = 1 - self.e_alpha_eta_mu
        p = p1 / (p1 + p2)

        def TP2_inv(u):
            return self.mu - (1.0/self.alpha) * jnp.log(1.0 - u * (1 - self.e_alpha_eta_mu))

        return jnp.where(u1 < p, self.TP1_inv(u2)[0], TP2_inv(u2))


    def tq_logpdf(self, x):
        """ Evaluate tildeq(x) \propto q(x) - min(p(x),q(x)) """
        Zq = 1 - self.e_alpha_eta_mu + self.e_alpha_gamma_mu - self.e_beta_gamma_eta
        # XXX: Note that this evaluates some logs of negative values which might cause trouble:
        return jnp.where((x >= self.eta) & (x <= self.gamma),
                         jnp.log(jnp.exp(texp_logpdf(x, self.eta, self.beta))
                                 - jnp.exp(texp_logpdf(x, self.mu, self.alpha))) - jnp.log(Zq), -jnp.inf)

    def tq_sample_icdf(self, key, N=1):
        """ Sample from tildeq(x) \propto q(x) - min(p(x),q(x)) """
        u = jax.random.uniform(key, shape=(N,))
        return self.TQ_inv(u)[0]


    def Gamma_hat(self, key, N=1, tp_rs=False, tq_rs=False):
        """ Sample from coupling of translated exponentials """
        # TODO: to be tested
        pxy = self.pxy()
        are_coupled = (u <= pxy)
        tp_samples = [] # TODO
        tq_samples = [] # TODO
        c_samples = []  # TODO
        xs = jnp.where(are_coupled, c_samples, tp_samples)
        ys = jnp.where(are_coupled, c_samples, tq_samples)
        return xs, ys, are_coupled

    def log_p(self, x):
        return -0.5 * (x - self.alpha) ** 2

    def log_q(self, x):
        return -0.5 * (x - self.beta) ** 2

    def log_p_hat(self, x):
        return jnp.ones_like(x)

    def log_q_hat(self, x):
        return jnp.ones_like(x)

    def M_p(self):
        # TODO
        pass

    def M_q(self):
        # TODO
        pass
