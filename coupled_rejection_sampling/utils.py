import math
from functools import partial

from jax import numpy as jnp

LOG_HALF = math.log(0.5)


@partial(jnp.vectorize, signature="(),()->()")
def logsubexp(x1, x2):
    amax = jnp.maximum(x1, x2)
    delta = jnp.abs(x1 - x2)
    return amax + log1mexp(-abs(delta))


def log1mexp(x):
    return jnp.where(x < LOG_HALF, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))
