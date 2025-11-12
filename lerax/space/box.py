from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from .base_space import AbstractSpace
from .utils import try_cast


class Box(AbstractSpace[Float[Array, " ..."]]):
    """
    A space of continuous values.

    A continuous closed set of floats.
    """

    _shape: tuple[int, ...]
    high: Float[Array, " ..."]
    low: Float[Array, " ..."]

    def __init__(
        self,
        low: Float[ArrayLike, " ..."],
        high: Float[ArrayLike, " ..."],
        shape: tuple[int, ...] | None = None,
    ):
        low = jnp.asarray(low, dtype=float)
        high = jnp.asarray(high, dtype=float)
        if shape is None:
            low, high = jnp.broadcast_arrays(low, high)
            shape = low.shape
            # TODO: Add warning if both shapes change

        assert low.shape == high.shape, "Box low and high must have the same shape"

        self._shape = shape
        self.low = jnp.broadcast_to(low, shape)
        self.high = jnp.broadcast_to(high, shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def canonical(self) -> Float[Array, " ..."]:
        return (self.low + self.high) / 2

    def sample(self, key: Key) -> Float[Array, " ..."]:
        bounded_key, unbounded_key, upper_bounded_key, lower_bounded_key = jr.split(
            key, 4
        )

        bounded_above = jnp.isfinite(self.high)
        bounded_below = jnp.isfinite(self.low)

        bounded = bounded_above & bounded_below
        unbounded = ~bounded_above & ~bounded_below
        upper_bounded = ~bounded_below & bounded_above
        lower_bounded = bounded_below & ~bounded_above

        sample = jnp.empty(self._shape, dtype=self.low.dtype)

        sample = jnp.where(
            bounded,
            jr.uniform(bounded_key, self._shape, minval=self.low, maxval=self.high),
            sample,
        )

        sample = jnp.where(unbounded, jr.normal(unbounded_key, self._shape), sample)

        sample = jnp.where(
            upper_bounded,
            self.high - jr.exponential(upper_bounded_key, self._shape),
            sample,
        )

        sample = jnp.where(
            lower_bounded,
            self.low + jr.exponential(lower_bounded_key, self._shape),
            sample,
        )

        return sample

    def contains(self, x: Any) -> Bool[Array, ""]:
        x = try_cast(x)
        if x is None:
            return jnp.array(False)

        if x.shape != self._shape:
            return jnp.array(False)

        return jnp.all(x >= self.low) & jnp.all(x <= self.high)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return False

        return bool(jnp.array_equal(self.low, other.low)) and bool(
            jnp.array_equal(self.high, other.high)
        )

    def __repr__(self) -> str:
        return f"Box(low={self.low}, high={self.high})"

    def __hash__(self) -> int:
        return hash((self.low.tobytes(), self.high.tobytes()))

    def flatten_sample(self, sample: Float[Array, " ..."]) -> Float[Array, " n"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_size(self) -> Int[Array, ""]:
        return jnp.prod(jnp.asarray(self._shape)).astype(int)

    @property
    def dtype(self) -> jnp.dtype:
        return self.low.dtype
