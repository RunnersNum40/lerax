from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Self

import equinox as eqx
from distreqx import bijectors, distributions
from jaxtyping import Array, Float, Key


class AbstractDistribution[SampleType](eqx.Module):
    """
    Base class for all distributions in Lerax.
    """

    @abstractmethod
    def log_prob(self, value: SampleType) -> Float[Array, ""]:
        """Compute the log probability of a sample."""

    @abstractmethod
    def prob(self, value: SampleType) -> Float[Array, ""]:
        """Compute the probability of a sample."""

    @abstractmethod
    def sample(self, key: Key[Array, ""]) -> SampleType:
        """Return a sample from the distribution."""

    @abstractmethod
    def entropy(self) -> Float[Array, ""]:
        """Compute the entropy of the distribution."""

    @abstractmethod
    def mean(self) -> SampleType:
        """Compute the mean of the distribution."""

    @abstractmethod
    def mode(self) -> SampleType:
        """Compute the mode of the distribution."""

    @abstractmethod
    def sample_and_log_prob(
        self, key: Key[Array, ""]
    ) -> tuple[SampleType, Float[Array, ""]]:
        """Return a sample and its log probability."""


class AbstractDistreqxWrapper[SampleType](AbstractDistribution):
    """
    Base class for distributions that wrap distreqx distributions in Lerax.

    Attributes:
        distribution: The underlying distreqx distribution.
    """

    distribution: eqx.AbstractVar[distributions.AbstractDistribution]

    def log_prob(self, value: SampleType) -> Float[Array, ""]:
        return self.distribution.log_prob(value)

    def prob(self, value: SampleType) -> Float[Array, ""]:
        return self.distribution.prob(value)

    def sample(self, key: Key[Array, ""]) -> SampleType:
        return self.distribution.sample(key)

    def entropy(self) -> Float[Array, ""]:
        return self.distribution.entropy()

    def mean(self) -> SampleType:
        return self.distribution.mean()

    def mode(self) -> SampleType:
        return self.distribution.mode()

    def sample_and_log_prob(
        self, key: Key[Array, ""]
    ) -> tuple[SampleType, Float[Array, ""]]:
        return self.distribution.sample_and_log_prob(key)


class AbstractMaskableDistribution[SampleType, MaskType](
    AbstractDistribution[SampleType]
):
    """
    Base class for all maskable distributions in Lerax.

    Maskable distributions allow masking of elements in the distribution.

    Attributes:
        distribution: The underlying distreqx distribution.
    """

    @abstractmethod
    def mask(self, mask: MaskType) -> Self:
        """
        Return a masked version of the distribution.

        A masked distribution only considers the elements where the mask is True.

        Args:
            mask: A mask indicating which elements to consider.

        Returns:
            A new masked distribution.
        """


class AbstractTransformedDistribution[SampleType](AbstractDistreqxWrapper[SampleType]):
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
            return self.distribution.bijector.forward(
                self.distribution.distribution.mode()
            )

    @property
    def bijector(self) -> bijectors.AbstractBijector:
        return self.distribution.bijector
