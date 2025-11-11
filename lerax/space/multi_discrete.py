from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from .base_space import AbstractSpace
from .utils import try_cast


class MultiDiscrete(AbstractSpace[Int[Array, " n"]]):
    """Cartesian product of discrete spaces."""

    ns: Int[Array, " n"]
    starts: Int[Array, " n"]

    def __init__(self, ns: tuple[int, ...], starts: tuple[int, ...] = (0,)):
        assert len(ns) > 0, "ns must be non-empty"
        starts = tuple(starts) if len(starts) > 0 else (0,) * len(ns)
        assert len(ns) == len(starts), "ns and starts must have the same length"
        assert all(n > 0 for n in ns), "all n must be positive"

        self.ns = jnp.array(ns, dtype=float)
        self.starts = jnp.array(starts, dtype=float)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.ns),)

    def canonical(self) -> Int[Array, " n"]:
        return self.starts

    def sample(self, key: Key) -> Int[Array, " n"]:
        return jr.randint(
            key, shape=self.shape, minval=self.starts, maxval=self.ns + self.starts
        )

    def contains(self, x: Any) -> Bool[Array, ""]:
        x = try_cast(x)
        if x is None:
            return jnp.array(False)

        if x.shape != self.shape:
            return jnp.array(False)

        if ~jnp.array_equal(x, jnp.floor(x)):
            return jnp.array(False)

        return jnp.all((self.starts <= x) & (x < self.ns + self.starts), axis=0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDiscrete):
            return False

        return bool(jnp.array_equal(self.ns, other.ns)) and bool(
            jnp.array_equal(self.starts, other.starts)
        )

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.ns}, starts={self.starts})"

    def __hash__(self) -> int:
        return hash((self.ns.tobytes(), self.starts.tobytes()))

    def flatten_sample(self, sample: Int[ArrayLike, " n"]) -> Float[Array, " size"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_size(self) -> Int[Array, ""]:
        return jnp.array(len(self.ns))
