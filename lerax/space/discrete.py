from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from .base_space import AbstractSpace
from .utils import try_cast


class Discrete(AbstractSpace[Int[Array, ""]]):
    """
    A space of finite discrete values.

    A finite closed set of integers.
    """

    _n: Int[Array, ""]
    start: Int[Array, ""]

    def __init__(self, n: Int[ArrayLike, ""], start: Int[ArrayLike, ""] = 0):
        assert n > 0, "n must be positive"  # pyright: ignore

        self._n = jnp.array(n, dtype=float)
        self.start = jnp.array(start, dtype=float)

    @property
    def n(self) -> Int[Array, ""]:
        return self._n

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def canonical(self) -> Int[Array, ""]:
        return self.start

    def sample(self, key: Key) -> Int[Array, ""]:
        return jr.randint(key, shape=(), minval=self.start, maxval=self._n + self.start)

    def contains(self, x: Any) -> Bool[Array, ""]:
        x = try_cast(x)
        if x is None:
            return jnp.array(False)

        if x.ndim != 0:
            return jnp.array(False)
        x = x.squeeze()

        if ~jnp.array_equal(x, jnp.floor(x)):
            return jnp.array(False)

        return self.start <= x < self._n + self.start

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Discrete):
            return False
        return bool((self._n == other._n) & (self.start == other.start))

    def __repr__(self) -> str:
        return f"Discrete({self._n}, start={self.start})"

    def __hash__(self) -> int:
        return hash((int(self._n), int(self.start)))

    def flatten_sample(self, sample: Int[Array, ""]) -> Float[Array, " 1"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_size(self) -> Int[Array, ""]:
        return jnp.array(1, dtype=int)

    @property
    def dtype(self) -> jnp.dtype:
        return self._n.dtype
