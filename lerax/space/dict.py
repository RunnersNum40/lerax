from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Int, Key

from .base_space import AbstractSpace


class Dict(AbstractSpace[dict[str, Any]]):
    """A dictionary of spaces."""

    spaces: dict[str, AbstractSpace]

    def __init__(self, spaces: dict[str, AbstractSpace]):
        assert isinstance(spaces, dict), "spaces must be a dict"
        assert len(spaces) > 0, "spaces must be non-empty"
        assert all(
            isinstance(space, AbstractSpace) for space in spaces.values()
        ), "spaces must be a dict of AbstractSpace"

        self.spaces = spaces

    @property
    def shape(self) -> None:
        return None

    def canonical(self) -> dict[str, Any]:
        return {key: space.canonical() for key, space in self.spaces.items()}

    def sample(self, key: Key) -> dict[str, Any]:
        return {
            space_key: self.spaces[space_key].sample(rng_key)
            for space_key, rng_key in zip(
                self.spaces.keys(), jr.split(key, len(self.spaces))
            )
        }

    def contains(self, x: Any) -> Bool[Array, ""]:
        if not isinstance(x, dict):
            return jnp.array(False)

        if len(x) != len(self.spaces):
            return jnp.array(False)

        return jnp.array(
            key in self.spaces and self.spaces[key].contains(x[key]) for key in x.keys()
        ).all()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dict):
            return False

        return all(
            key in other.spaces and self.spaces[key] == other.spaces[key]
            for key in self.spaces.keys()
        )

    def __repr__(self) -> str:
        return f"Dict({', '.join(f'{key}: {repr(space)}' for key, space in self.spaces.items())})"

    def __hash__(self) -> int:
        return hash(tuple((key, hash(space)) for key, space in self.spaces.items()))

    def flatten_sample(self, sample: dict[str, Any]) -> Float[Array, " size"]:
        parts = [
            subspace.flatten_sample(sample[key])
            for key, subspace in sorted(self.spaces.items())
        ]
        return jnp.concatenate(parts)

    @property
    def flat_size(self) -> Int[Array, ""]:
        return jnp.array(space.flat_size for space in self.spaces.values()).sum()

    def __getitem__(self, index: str) -> AbstractSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)
