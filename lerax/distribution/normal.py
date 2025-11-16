from __future__ import annotations

from distreqx import distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base_distribution import AbstractDistribution


class Normal(AbstractDistribution[Float[Array, " dims"]]):

    distribution: distributions.Normal

    def __init__(
        self,
        loc: Float[ArrayLike, " dims"],
        scale: Float[ArrayLike, " dims"],
    ):
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)

        if loc.shape != scale.shape:
            raise ValueError("loc and scale must have the same shape.")

        self.distribution = distributions.Normal(loc=loc, scale=scale)

    @property
    def loc(self) -> Float[Array, " dims"]:
        return self.distribution.loc

    @property
    def scale(self) -> Float[Array, " dims"]:
        return self.distribution.scale
