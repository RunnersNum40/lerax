from __future__ import annotations

from distreqx import bijectors, distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from .base_distribution import (
    AbstractDistribution,
    AbstractMaskableDistribution,
    AbstractTransformedDistribution,
)


class Bernoulli(AbstractMaskableDistribution[Bool[Array, " dims"]]):

    distribution: distributions.Bernoulli

    def __init__(
        self,
        logits: Float[ArrayLike, " dims"] | None = None,
        probs: Float[ArrayLike, " dims"] | None = None,
    ):
        logits = jnp.asarray(logits) if logits is not None else None
        probs = jnp.asarray(probs) if probs is not None else None
        self.distribution = distributions.Bernoulli(logits=logits, probs=probs)

    @property
    def logits(self) -> Float[Array, " dims"]:
        return self.distribution.logits

    @property
    def probs(self) -> Float[Array, " dims"]:
        return self.distribution.probs

    def mask(self, mask: Bool[Array, " dims"]) -> Bernoulli:
        masked_logits = jnp.where(mask, self.logits, -jnp.inf)
        return Bernoulli(logits=masked_logits)


class Categorical(AbstractMaskableDistribution[Integer[Array, ""]]):

    distribution: distributions.Categorical

    def __init__(
        self,
        logits: Float[ArrayLike, " dims"] | None = None,
        probs: Float[ArrayLike, " dims"] | None = None,
    ):
        logits = jnp.asarray(logits) if logits is not None else None
        probs = jnp.asarray(probs) if probs is not None else None

        self.distribution = distributions.Categorical(logits=logits, probs=probs)

    @property
    def logits(self) -> Float[Array, " dims"]:
        return self.distribution.logits

    @property
    def probs(self) -> Float[Array, " dims"]:
        return self.distribution.probs

    def mask(self, mask: Bool[Array, " dims"]) -> Categorical:
        masked_logits = jnp.where(mask, self.logits, -jnp.inf)
        return Categorical(logits=masked_logits)


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


class MultivariateNormalDiag(AbstractDistribution[Float[Array, " dims"]]):

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


class SquashedMultivariateNormalDiag(
    AbstractTransformedDistribution[Float[Array, " dims"]]
):
    """Multivariate Normal with squashing bijector for bounded outputs."""

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
