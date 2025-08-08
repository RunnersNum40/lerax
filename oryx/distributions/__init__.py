"""
Oryx Distributions

Wrapper around distreqx.distributions to allow for easier imports, extended typing, and
future expansion.
"""

from __future__ import annotations

import equinox as eqx
from distreqx import bijectors, distributions
from jaxtyping import Array, Bool, Float, Integer, Key


class AbstractDistribution[SampleType](eqx.Module):
    """Base class for all distributions in Oryx."""

    distribution: eqx.AbstractVar[distributions.AbstractDistribution]

    def log_prob(self, value: SampleType) -> Float[Array, ""]:
        """Compute the log probability of a sample."""
        return self.distribution.log_prob(value)

    def prob(self, value: SampleType) -> Float[Array, ""]:
        """Compute the probability of a sample."""
        return self.distribution.prob(value)

    def sample(self, key: Key) -> SampleType:
        """Return a sample from the distribution."""
        return self.distribution.sample(key)

    def entropy(self) -> Float[Array, ""]:
        """Compute the entropy of the distribution."""
        return self.distribution.entropy()

    def mean(self) -> SampleType:
        """Compute the mean of the distribution."""
        return self.distribution.mean()

    def mode(self) -> SampleType:
        """Compute the mode of the distribution."""
        return self.distribution.mode()

    def sample_and_log_prob(self, key: Key) -> tuple[SampleType, Float[Array, ""]]:
        """Return a sample and its log probability."""
        return self.distribution.sample_and_log_prob(key)


class AbstractTransformedDistribution[SampleType](AbstractDistribution[SampleType]):

    distribution: eqx.AbstractVar[distributions.AbstractTransformed]

    @property
    def bijector(self) -> bijectors.AbstractBijector:
        return self.distribution.bijector


class Bernoulli(AbstractDistribution[Bool[Array, " dims"]]):

    distribution: distributions.Bernoulli

    def __init__(
        self,
        logits: Float[Array, " dims"] | None = None,
        probs: Float[Array, " dims"] | None = None,
    ):
        self.distribution = distributions.Bernoulli(logits=logits, probs=probs)

    @property
    def logits(self) -> Float[Array, " dims"]:
        return self.distribution.logits

    @property
    def probs(self) -> Float[Array, " dims"]:
        return self.distribution.probs


class Categorical(AbstractDistribution[Integer[Array, ""]]):

    distribution: distributions.Categorical

    def __init__(
        self,
        logits: Float[Array, " dims"] | None = None,
        probs: Float[Array, " dims"] | None = None,
    ):
        self.distribution = distributions.Categorical(logits=logits, probs=probs)

    @property
    def logits(self) -> Float[Array, " dims"]:
        return self.distribution.logits

    @property
    def probs(self) -> Float[Array, " dims"]:
        return self.distribution.probs


class Normal(AbstractDistribution[Float[Array, " dims"]]):

    distribution: distributions.Normal

    def __init__(
        self,
        loc: Float[Array, " dims"],
        scale: Float[Array, " dims"],
    ):
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

    def __init__(self, loc: Float[Array, " dims"], scale: Float[Array, " dims"]):
        if loc.shape != scale.shape:
            raise ValueError("loc and scale must have the same shape.")

        normal = distributions.Normal(loc=loc, scale=scale)
        tanh = bijectors.Tanh()

        self.distribution = distributions.Transformed(normal, tanh)


class MultivariateNormalDiag(AbstractDistribution[Float[Array, " dims"]]):

    distribution: distributions.MultivariateNormalDiag

    def __init__(
        self,
        loc: Float[Array, " dims"] | None = None,
        scale_diag: Float[Array, " dims"] | None = None,
    ):
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
        loc: Float[Array, " dims"],
        scale_diag: Float[Array, " dims"],
        high: Float[Array, " dims"] | None = None,
        low: Float[Array, " dims"] | None = None,
    ):
        """
        Initialize a SquashedMultivariateNormalDiag distribution.

        Either both high and low must be provided for bounded squashing or neither.
        If neither are provided, the distribution will use a Tanh bijector for squashing
        between -1 and 1.
        """
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


__all__ = [
    "AbstractDistribution",
    "Bernoulli",
    "Categorical",
    "Normal",
    "MultivariateNormalDiag",
    "SquashedMultivariateNormalDiag",
]
