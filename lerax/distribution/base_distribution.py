from __future__ import annotations

import warnings
from abc import abstractmethod

import equinox as eqx
from distreqx import bijectors, distributions
from jaxtyping import Array, Bool, Float, Key


class AbstractDistribution[SampleType](eqx.Module):
    """
    Base class for all distributions in Lerax.

    Lerax distributions wrap around `distreqx` distributions to provide a
    convenient interface for reinforcement learning.

    Attributes:
        distribution: The underlying distreqx distribution.
    """

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
    """
    Base class for all maskable distributions in Lerax.

    Maskable distributions allow masking of elements in the distribution.

    Attributes:
        distribution: The underlying distreqx distribution.
    """

    distribution: eqx.AbstractVar[distributions.AbstractDistribution]

    @abstractmethod
    def mask[SelfType](self: SelfType, mask: Bool[Array, "..."]) -> SelfType:
        """
        Return a masked version of the distribution.

        A masked distribution only considers the elements where the mask is True.

        Args:
            mask: A boolean array indicating which elements to keep.

        Returns:
            A new masked distribution.
        """


class AbstractTransformedDistribution[SampleType](AbstractDistribution[SampleType]):
    """
    Base class for all transformed distributions in Lerax.

    Transformed distributions apply a bijective transformation to a base distribution.

    Attributes:
        distribution: The underlying distreqx transformed distribution.
        bijector: The bijective transformation applied to the base distribution.
    """

    distribution: eqx.AbstractVar[distributions.AbstractTransformed]

    # This breaks from the abstract/formal pattern but I think it's justified
    def mode(self) -> SampleType:
        try:
            return self.distribution.mode()
        except NotImplementedError:
            # Computing the mode this way is not always correct, but it is a reasonable workaround for the
            # use cases of this library.
            warnings.warn(
                "Mode not implemented for base distribution; using bijector to compute mode."
            )
            return self.distribution._bijector.forward(
                self.distribution._distribution.mode()
            )

    @property
    def bijector(self) -> bijectors.AbstractBijector:
        return self.distribution.bijector
