from __future__ import annotations

import equinox as eqx
from distreqx import bijectors, distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base_distribution import AbstractTransformedDistribution


class SquashedMultivariateNormalDiag(
    AbstractTransformedDistribution[Float[Array, " dims"]]
):
    """
    Multivariate Normal with squashing bijector for bounded outputs.

    Attributes:
        distribution: The underlying distreqx Transformed distribution.

    Args:
        loc: The mean of the multivariate normal distribution.
        scale_diag: The diagonal of the covariance matrix.
        high: The upper bound for bounded squashing..
        low: The lower bound for bounded squashing..
    """

    distribution: distributions.Transformed

    def __init__(
        self,
        loc: Float[ArrayLike, " dims"],
        scale_diag: Float[ArrayLike, " dims"],
        high: Float[ArrayLike, " dims"] = jnp.array(1.0),
        low: Float[ArrayLike, " dims"] = jnp.array(-1.0),
    ):
        loc = jnp.asarray(loc)
        scale_diag = jnp.asarray(scale_diag)

        (high, low) = eqx.error_if(
            (high, low),
            ~(jnp.isfinite(high) & jnp.isfinite(low)),
            "SquashedMultivariateNormalDiag requires finite low/high for all "
            "dimensions. Got non-finite bounds.",
        )

        high = jnp.broadcast_to(jnp.asarray(high), loc.shape)
        low = jnp.broadcast_to(jnp.asarray(low), loc.shape)

        mvn = distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        sigmoid = bijectors.Sigmoid()
        affine = bijectors.ScalarAffine(scale=(high - low), shift=low)
        chain = bijectors.Chain((affine, sigmoid))
        bijector = bijectors.Block(chain, ndims=1)

        self.distribution = distributions.Transformed(mvn, bijector)

    @property
    def loc(self) -> Float[Array, " dims"]:
        assert isinstance(
            self.distribution._distribution, distributions.MultivariateNormalDiag
        )
        return self.distribution._distribution.loc

    @property
    def scale_diag(self) -> Float[Array, " dims"]:
        assert isinstance(
            self.distribution._distribution, distributions.MultivariateNormalDiag
        )
        return self.distribution._distribution.scale_diag
