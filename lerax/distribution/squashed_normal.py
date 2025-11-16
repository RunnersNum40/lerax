from __future__ import annotations

from distreqx import bijectors, distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base_distribution import AbstractTransformedDistribution


class SquashedNormal(AbstractTransformedDistribution[Float[Array, " dims"]]):

    distribution: distributions.Transformed

    def __init__(
        self,
        loc: Float[ArrayLike, " dims"],
        scale: Float[ArrayLike, " dims"],
        high: Float[ArrayLike, " dims"] | None = None,
        low: Float[ArrayLike, " dims"] | None = None,
    ):
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        high = jnp.asarray(high) if high is not None else None
        low = jnp.asarray(low) if low is not None else None

        if loc.shape != scale.shape:
            raise ValueError("loc and scale must have the same shape.")

        normal = distributions.Normal(loc=loc, scale=scale)

        if high is not None or low is not None:
            assert (
                high is not None and low is not None
            ), "Both high and low must be provided for bounded squashing."
            sigmoid = bijectors.Sigmoid()
            affine = bijectors.ScalarAffine(scale=high - low, shift=low)
            chain = bijectors.Chain((sigmoid, affine))
            self.distribution = distributions.Transformed(normal, chain)
        else:
            tanh = bijectors.Tanh()
            self.distribution = distributions.Transformed(normal, tanh)

    @property
    def loc(self) -> Float[Array, " dims"]:
        assert isinstance(self.distribution._distribution, distributions.Normal)
        return self.distribution._distribution.loc

    @property
    def scale(self) -> Float[Array, " dims"]:
        assert isinstance(self.distribution._distribution, distributions.Normal)
        return self.distribution._distribution.scale
