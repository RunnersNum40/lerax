from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from distreqx import bijectors, distributions
from jaxtyping import Array, Bool, Float, Key


class AbstractDistribution[SampleType](eqx.Module):
    """Base class for all distributions in Lerax."""

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


class AbstractMaskableDistribution[SampleType](AbstractDistribution[SampleType]):

    distribution: eqx.AbstractVar[distributions.AbstractDistribution]

    @abstractmethod
    def mask[SelfType](self: SelfType, mask: Bool[Array, "..."]) -> SelfType:
        """Return a masked version of the distribution."""


class AbstractTransformedDistribution[SampleType](AbstractDistribution[SampleType]):

    distribution: eqx.AbstractVar[distributions.AbstractTransformed]

    def mode(self) -> SampleType:
        # Computing the mode this way is not always correct, but it is a reasonable workaround for the
        # use cases of this library.
        try:
            return super().mode()
        except NotImplementedError:
            # TODO: Add a warning here about the mode not being implemented for
            # the underlying distribution.
            return self.distribution._bijector.forward(
                self.distribution._distribution.mode()
            )

    @property
    def bijector(self) -> bijectors.AbstractBijector:
        return self.distribution.bijector
