from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from .base_space import AbstractSpace


class Tuple(AbstractSpace[tuple[Any, ...], None]):
    """
    A cartesian product of spaces.

    Attributes:
        spaces: A tuple of the component spaces.

    Args:
        spaces: A tuple of the component spaces.
    """

    spaces: tuple[AbstractSpace, ...]

    def __init__(self, spaces: tuple[AbstractSpace, ...]):
        assert isinstance(spaces, tuple), "spaces must be a tuple"
        assert len(spaces) > 0, "spaces must be non-empty"
        assert all(isinstance(space, AbstractSpace) for space in spaces), (
            "spaces must be a tuple of AbstractSpace"
        )

        self.spaces = spaces

    @property
    def shape(self) -> None:
        return None

    def canonical(self) -> tuple[Any, ...]:
        return tuple(space.canonical() for space in self.spaces)

    def sample(self, *, key: Key[Array, ""], mask: None = None) -> tuple[Any, ...]:
        """
        Returns a random sample from the space.

        Args:
            key: A JAX PRNGKey[Array, ""].

        Returns:
            A tuple containing a random sample from each component space.
        """
        return tuple(
            space.sample(key=key)
            for space, key in zip(self.spaces, jr.split(key, len(self.spaces)))
        )

    def contains(self, x: Any) -> Bool[Array, ""]:
        """
        Returns whether the given value is contained in the space.

        Args:
            x: The value to check.

        Returns:
            Whether the value is contained in the space.
        """
        if not isinstance(x, tuple):
            return jnp.array(False)

        if len(x) != len(self.spaces):
            return jnp.array(False)

        return jnp.array(
            [space.contains(x_i) for space, x_i in zip(self.spaces, x)]
        ).all()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tuple):
            return False

        return all(
            space == other_space
            for space, other_space in zip(self.spaces, other.spaces)
        )

    def __repr__(self) -> str:
        return f"Tuple({', '.join(repr(space) for space in self.spaces)})"

    def __hash__(self) -> int:
        return hash(tuple(hash(space) for space in self.spaces))

    def flatten_sample(self, sample: tuple[Any, ...]) -> Float[Array, " size"]:
        parts = [
            subspace.flatten_sample(subsample)
            for subsample, subspace in zip(sample, self.spaces)
        ]
        return jnp.concatenate(parts)

    @property
    def flat_size(self) -> int:
        return sum(space.flat_size for space in self.spaces)

    def __getitem__(self, index: int) -> AbstractSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)
