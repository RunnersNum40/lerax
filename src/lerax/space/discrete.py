from __future__ import annotations

from typing import Any

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from .base_space import AbstractSpace
from .utils import try_cast


class Discrete(AbstractSpace[Int[Array, ""], Bool[Array, " n"]]):
    """
    A space of finite discrete values.

    Attributes:
        n: The number of discrete values.

    Args:
        n: The number of discrete values.
    """

    n: int = eqx.field(static=True)

    def __init__(self, n: int):
        assert n > 0, "n must be positive"

        self.n = n

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def canonical(self) -> Int[Array, ""]:
        return jnp.array(0, dtype=int)

    def sample(
        self, *, mask: Bool[ArrayLike, " n"] | None = None, key: Key[Array, ""]
    ) -> Int[Array, ""]:
        mask = (
            jnp.asarray(mask) if mask is not None else jnp.ones((self.n,), dtype=bool)
        )
        return jr.choice(key, self.n, p=mask / jnp.sum(mask))

    def contains(self, x: Any) -> Bool[Array, ""]:
        x = try_cast(x)
        if x is None:
            return jnp.array(False)

        if x.ndim != 0:
            return jnp.array(False)
        x = x.squeeze()

        if ~jnp.array_equal(x, jnp.floor(x)):
            return jnp.array(False)

        return 0 <= x < self.n

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"

    def __hash__(self) -> int:
        return hash(self.n)

    def flatten_sample(self, sample: Int[Array, ""]) -> Float[Array, " 1"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_size(self) -> int:
        return 1
