from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jaxtyping import Array, Bool, Float, Key


class AbstractSpace[SampleType](eqx.Module):
    """
    Abstract base class for defining a space.

    A space is a set of values that can be sampled from.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...] | None:
        """The shape of a sample of the space."""

    @abstractmethod
    def sample(self, key: Key) -> SampleType:
        """Returns a random sample from the space."""

    @abstractmethod
    def contains(self, x: Any) -> Bool[Array, ""]:
        """Returns True if the input is in the space, False otherwise."""

    @abstractmethod
    def flatten_sample(self, sample: SampleType) -> Float[Array, " n"]:
        """Flatten a sample from the space into a 1-D array."""

    @property
    @abstractmethod
    def flat_size(self) -> int:
        """Return the size of a flattened sample from the space."""

    @abstractmethod
    def canonical(self) -> SampleType:
        """Return an element of the space."""

    def __contains__(self, x: Any) -> bool:
        return bool(self.contains(x))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Return if this space is equal to another space."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the space."""

    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash of the space."""
