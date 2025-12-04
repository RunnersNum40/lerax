from __future__ import annotations

from distreqx import bijectors, distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base_distribution import AbstractTransformedDistribution


class SquashedMultivariateNormalDiag(
    AbstractTransformedDistribution[Float[Array, " dims"]]
):
    """
    Multivariate Normal with squashing bijector for bounded outputs.

    Note:
        Either both `high` and `low` must be provided for bounded squashing,
        or neither should be provided for tanh squashing.

    Attributes:
        distribution: The underlying distreqx Transformed distribution.

    Args:
        loc: The mean of the multivariate normal distribution.
        scale_diag: The diagonal of the covariance matrix.
        high: The upper bound for bounded squashing. If None, uses tanh squashing.
        low: The lower bound for bounded squashing. If None, uses tanh squashing.
    """

    distribution: distributions.Transformed

    def __init__(
        self,
        loc: Float[ArrayLike, " dims"],
        scale_diag: Float[ArrayLike, " dims"],
        high: Float[ArrayLike, " dims"] | None = None,
        low: Float[ArrayLike, " dims"] | None = None,
    ):
        """
        Initialize a SquashedMultivariateNormalDiag distribution.

        Either both high and low must be provided for bounded squashing or neither.
        If neither are provided, the distribution will use a Tanh bijector for squashing
        between -1 and 1.
        """
        loc = jnp.asarray(loc)
        scale_diag = jnp.asarray(scale_diag)
        high = jnp.asarray(high) if high is not None else None
        low = jnp.asarray(low) if low is not None else None

        mvn = distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        if high is not None or low is not None:
            assert (
                high is not None and low is not None
            ), "Both high and low must be provided for bounded squashing."

            sigmoid = bijectors.Sigmoid()
            scale = bijectors.DiagLinear(high - low)
            shift = bijectors.Shift(low)
            chain = bijectors.Chain((sigmoid, scale, shift))
            self.distribution = distributions.Transformed(mvn, chain)
        else:
            tanh = bijectors.Tanh()
            self.distribution = distributions.Transformed(mvn, tanh)

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
