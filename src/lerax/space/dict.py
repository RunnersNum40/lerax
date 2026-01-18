from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from .base_space import AbstractSpace


class Dict(AbstractSpace[OrderedDict[str, Any], None]):
    """
    A dictionary of spaces.

    Attributes:
        spaces: An ordered dictionary mapping keys to component spaces.

    Args:
        spaces: A mapping from keys to component spaces.
    """

    spaces: OrderedDict[str, AbstractSpace]

    def __init__(self, spaces: Mapping[str, AbstractSpace]):
        self.spaces = OrderedDict(spaces)

    @property
    def shape(self) -> None:
        return None

    def canonical(self) -> OrderedDict[str, Any]:
        return OrderedDict(
            {key: space.canonical() for key, space in self.spaces.items()}
        )

    def sample(self, *, key: Key, mask: None = None) -> OrderedDict[str, Any]:
        return OrderedDict(
            {
                space_key: space.sample(key=subkey)
                for space_key, space, subkey in zip(
                    self.spaces.keys(),
                    self.spaces.values(),
                    jr.split(key, len(self.spaces)),
                )
            }
        )

    def contains(self, x: Any) -> Bool[Array, ""]:
        if not isinstance(x, OrderedDict):
            return jnp.array(False)

        if self.spaces.keys() != x.keys():
            return jnp.array(False)

        return jnp.array(
            [space.contains(x[key]) for key, space in self.spaces.items()]
        ).all()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrderedDict):
            return False

        return all(
            self_key == other_key and self_value == other_value
            for (self_key, self_value), (other_key, other_value) in zip(
                self.spaces.items(), other.items()
            )
        )

    def __repr__(self) -> str:
        return f"Dict({', '.join(f'{key}: {repr(space)}' for key, space in self.spaces.items())})"

    def __hash__(self) -> int:
        return hash(self.spaces.items())

    def flatten_sample(self, sample: OrderedDict[str, Any]) -> Float[Array, " size"]:
        parts = [
            space.flatten_sample(sample[key]) for key, space in self.spaces.items()
        ]
        return jnp.concatenate(parts)

    @property
    def flat_size(self) -> int:
        return sum(space.flat_size for space in self.spaces.values())

    def __getitem__(self, index: str) -> AbstractSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)
