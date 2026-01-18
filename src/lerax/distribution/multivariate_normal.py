from __future__ import annotations

from distreqx import distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base_distribution import AbstractDistreqxWrapper


class MultivariateNormalDiag(AbstractDistreqxWrapper[Float[Array, " dims"]]):
    """
    Multivariate Normal distribution with diagonal covariance.

    Attributes:
        distribution: The underlying distreqx MultivariateNormalDiag distribution.

    Args:
        loc: The mean of the distribution.
        scale_diag: The diagonal of the covariance matrix.
    """

    distribution: distributions.MultivariateNormalDiag

    def __init__(
        self,
        loc: Float[ArrayLike, " dims"] | None = None,
        scale_diag: Float[ArrayLike, " dims"] | None = None,
    ):
        loc = jnp.asarray(loc) if loc is not None else None
        scale_diag = jnp.asarray(scale_diag) if scale_diag is not None else None

        if (loc is not None and scale_diag is not None) and (
            loc.shape != scale_diag.shape
        ):
            raise ValueError("loc and scale_diag must have the same shape.")

        self.distribution = distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag
        )

    @property
    def loc(self) -> Float[Array, " dims"]:
        return self.distribution.loc

    @property
    def scale_diag(self) -> Float[Array, " dims"]:
        return self.distribution.scale_diag
