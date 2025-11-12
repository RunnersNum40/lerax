from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int, Key


class AbstractSpace[SampleType](eqx.Module):
    """
    Abstract base class for defining a space.

    A space is a set of values that can be sampled from.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...] | None:
        """Returns the shape of the space as an immutable property."""

    @abstractmethod
    def canonical(self) -> SampleType:
        """Returns a canonical element of the space."""

    @abstractmethod
    def sample(self, key: Key) -> SampleType:
        """Returns a random sample from the space."""

    @abstractmethod
    def contains(self, x: Any) -> Bool[Array, ""]:
        """Returns True if the input is in the space, False otherwise."""

    def __contains__(self, x: Any) -> bool:
        return bool(self.contains(x))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Checks if two spaces are equal based on their properties."""

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the space."""

    @abstractmethod
    def __hash__(self) -> int:
        """Returns a hash of the space."""

    @abstractmethod
    def flatten_sample(self, sample: SampleType) -> Float[Array, " n"]:
        """Flattens a sample from the space into a 1-D array."""

    @property
    @abstractmethod
    def flat_size(self) -> Int[Array, ""]:
        """Returns the dimension of the flattened sample."""
