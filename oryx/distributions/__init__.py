"""
Oryx Distributions

Wrapper around distreqx.distributions to allow for easier imports, extended typing, and
future expansion.
"""

import equinox as eqx
from distreqx import bijectors, distributions
from jaxtyping import Array, Bool, Float, Int, Key


class AbstractDistribution[SampleType](eqx.Module, strict=True):
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


class AbstractSampleLogProbDistribution[SampleType](
    AbstractDistribution[SampleType], strict=True
):

    distribution: eqx.AbstractVar[distributions.AbstractSampleLogProbDistribution]

    def sample_and_log_prob(self, key: Key) -> tuple[SampleType, Float[Array, ""]]:
        """Return a sample and its log probability."""
        return self.distribution.sample_and_log_prob(key)


class AbstractTransformedDistribution[SampleType](
    AbstractDistribution[SampleType], strict=True
):

    distribution: eqx.AbstractVar[distributions.AbstractTransformed]

    @property
    def bijector(self) -> bijectors.AbstractBijector:
        return self.distribution.bijector

    def sample_and_log_prob(self, key: Key) -> tuple[SampleType, Float[Array, ""]]:
        """Return a sample and its log probability."""
        # TODO: Fix typing for sample_and_log_prob
        return self.distribution.sample_and_log_prob(key)  # pyright: ignore


class Bernoulli(AbstractSampleLogProbDistribution[Bool[Array, " dims"]], strict=True):

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


class Categorical(AbstractSampleLogProbDistribution[Int[Array, ""]], strict=True):

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


class Normal(AbstractSampleLogProbDistribution[Float[Array, " dims"]], strict=True):

    distribution: distributions.Normal

    def __init__(
        self,
        loc: Float[Array, " dims"],
        scale: Float[Array, " dims"],
    ):
        self.distribution = distributions.Normal(loc=loc, scale=scale)

    @property
    def loc(self) -> Float[Array, " dims"]:
        return self.distribution.loc

    @property
    def scale(self) -> Float[Array, " dims"]:
        return self.distribution.scale


class SquashedNormal(
    AbstractTransformedDistribution[Float[Array, " dims"]], strict=True
):

    distribution: distributions.Transformed

    def __init__(self, loc: Float[Array, " dims"], scale: Float[Array, " dims"]):
        normal = distributions.Normal(loc=loc, scale=scale)
        tanh = bijectors.Tanh()

        self.distribution = distributions.Transformed(normal, tanh)


class MultivariateNormalDiag(AbstractDistribution[Float[Array, " dims"]], strict=True):

    distribution: distributions.MultivariateNormalDiag

    def __init__(
        self,
        loc: Float[Array, " dims"] | None = None,
        scale_diag: Float[Array, " dims"] | None = None,
    ):
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
    AbstractTransformedDistribution[Float[Array, " dims"]], strict=True
):

    distribution: distributions.Transformed

    def __init__(
        self,
        loc: Float[Array, " dims"],
        scale_diag: Float[Array, " dims"],
    ):
        mvn = distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        tanh = bijectors.Tanh()

        self.distribution = distributions.Transformed(mvn, tanh)


__all__ = [
    "AbstractDistribution",
    "AbstractSampleLogProbDistribution",
    "Bernoulli",
    "Categorical",
    "Normal",
    "MultivariateNormalDiag",
]
