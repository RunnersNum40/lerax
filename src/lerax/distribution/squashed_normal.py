from __future__ import annotations

import equinox as eqx
from distreqx import bijectors, distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base_distribution import AbstractTransformedDistribution


class SquashedNormal(AbstractTransformedDistribution[Float[Array, " dims"]]):
    """
    Normal distribution with squashing bijector for bounded outputs.

    Attributes:
        distribution: The underlying distreqx Transformed distribution.

    Args:
        loc: The mean of the normal distribution.
        scale: The standard deviation of the normal distribution.
        high: The upper bound for bounded squashing.
        low: The lower bound for bounded squashing.
    """

    distribution: distributions.Transformed

    def __init__(
        self,
        loc: Float[ArrayLike, ""],
        scale: Float[ArrayLike, ""],
        high: Float[ArrayLike, ""] = jnp.array(1.0),
        low: Float[ArrayLike, ""] = jnp.array(-1.0),
    ):
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)

        (high, low) = eqx.error_if(
            (high, low),
            ~(jnp.isfinite(high) & jnp.isfinite(low)),
            "SquashedNormal requires finite low/high for all "
            "dimensions. Got non-finite bounds.",
        )

        high = jnp.asarray(high)
        low = jnp.asarray(low)

        normal = distributions.Normal(loc=loc, scale=scale)

        sigmoid = bijectors.Sigmoid()
        affine = bijectors.ScalarAffine(scale=(high - low), shift=low)
        bijector = bijectors.Chain((affine, sigmoid))

        self.distribution = distributions.Transformed(normal, bijector)

    @property
    def loc(self) -> Float[Array, " dims"]:
        assert isinstance(self.distribution._distribution, distributions.Normal)
        return self.distribution._distribution.loc

    @property
    def scale(self) -> Float[Array, " dims"]:
        assert isinstance(self.distribution._distribution, distributions.Normal)
        return self.distribution._distribution.scale
