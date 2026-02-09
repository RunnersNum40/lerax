from __future__ import annotations

from typing import Any

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from .base_space import AbstractSpace
from .utils import try_cast


class MultiDiscrete(AbstractSpace[Int[Array, " n"], None]):
    """
    Cartesian product of discrete spaces.

    Attributes:
        nvec: The number of discrete values for each dimension.

    Args:
        nvec: The number of discrete values for each dimension.
    """

    nvec: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, nvec: tuple[int, ...]):
        assert len(nvec) > 0, "nvec must be non-empty"
        assert all(n > 0 for n in nvec), "all n must be positive"

        self.nvec = nvec

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.nvec),)

    def canonical(self) -> Int[Array, " n"]:
        return jnp.zeros(self.shape, dtype=int)

    def sample(self, *, key: Key[Array, ""], mask: None = None) -> Int[Array, " n"]:
        return jr.randint(
            key,
            shape=self.shape,
            minval=0,
            maxval=jnp.array(self.nvec, dtype=int),
            dtype=int,
        )

    def contains(self, x: Any) -> Bool[Array, ""]:
        x = try_cast(x)
        if x is None:
            return jnp.array(False)

        if x.shape != self.shape:
            return jnp.array(False)

        if ~jnp.array_equal(x, jnp.floor(x)):
            return jnp.array(False)

        return jnp.all(x < jnp.asarray(self.nvec))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDiscrete):
            return False

        return self.nvec == other.nvec

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.nvec})"

    def __hash__(self) -> int:
        return hash(self.nvec)

    def flatten_sample(self, sample: Int[ArrayLike, " n"]) -> Float[Array, " n"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_size(self) -> int:
        return len(self.nvec)
