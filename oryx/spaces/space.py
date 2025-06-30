from abc import abstractmethod
from typing import Any

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key


class AbstractSpace[T](eqx.Module, strict=True):
    """
    Abstract base class for defining a space.

    A space is a set of values that can be sampled from.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...] | None:
        """Returns the shape of the space as an immutable property."""

    @abstractmethod
    def sample(self, key: Key) -> T:
        """Returns a random sample from the space."""

    @abstractmethod
    def contains(self, x: Any) -> Bool[Array, ""]:
        """Returns True if the input is in the space, False otherwise."""

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the space."""


class Discrete(AbstractSpace[Int[Array, ""]], strict=True):
    """
    A space of finite discrete values.

    A finite closed set of integers.
    """

    _n: Int[Array, ""]
    start: Int[Array, ""]

    def __init__(self, n: int, start: int = 0):
        assert n > 0, "n must be positive"

        self._n = jnp.asarray(n)
        self.start = jnp.asarray(start)

    @property
    def n(self) -> Int[Array, ""]:
        return self._n

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def sample(self, key: Key) -> Int[Array, ""]:
        return jr.randint(key, shape=(), minval=self.start, maxval=self._n + self.start)

    def contains(self, x: Any) -> Bool[Array, ""]:
        # TODO: Add type checking asserts
        return self.start <= x < self._n + self.start

    def __repr__(self) -> str:
        return f"Discrete({self._n}, start={self.start})"


class Box(AbstractSpace[Float[Array, " ..."]], strict=True):
    """
    A space of continuous values.

    A continuous closed set of floats.
    """

    _shape: tuple[int, ...]
    _high: Float[Array, " ..."]
    _low: Float[Array, " ..."]

    def __init__(
        self,
        low: Float[ArrayLike, " ..."],
        high: Float[ArrayLike, " ..."],
        shape: tuple[int, ...] | None,
    ):
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        if shape is None:
            low = jnp.broadcast_to(low, high.shape)
            high = jnp.broadcast_to(high, low.shape)
            shape = low.shape

        assert low.shape == high.shape, "low and high must have the same shape"

        self._shape = shape
        self._low = low
        self._high = high

    @property
    def low(self) -> Float[Array, " ..."]:
        return self._low

    @property
    def high(self) -> Float[Array, " ..."]:
        return self._high

    @property
    def shape(self) -> tuple[int, ...]:
        return self._low.shape

    def sample(self, key: Key) -> Float[Array, " ..."]:
        bounded_key, unbounded_key, upper_bounded_key, lower_bounded_key = jr.split(
            key, 4
        )

        bounded_above = jnp.isfinite(self._high)
        bounded_below = jnp.isfinite(self._low)

        bounded = bounded_above & bounded_below
        unbounded = ~bounded_above & ~bounded_below
        upper_bounded = ~bounded_below & bounded_above
        lower_bounded = bounded_below & ~bounded_above

        sample = jnp.empty(self.shape, dtype=self._low.dtype)

        sample = jnp.where(
            bounded,
            jr.uniform(bounded_key, self.shape, minval=self._low, maxval=self._high),
            sample,
        )

        sample = jnp.where(unbounded, jr.normal(unbounded_key, self.shape), sample)

        sample = jnp.where(
            upper_bounded,
            self._high - jr.exponential(upper_bounded_key, self.shape),
            sample,
        )

        sample = jnp.where(
            lower_bounded,
            self._low + jr.exponential(lower_bounded_key, self.shape),
            sample,
        )

        return sample

    def contains(self, x: Any) -> Bool[Array, ""]:
        assert isinstance(x, jnp.ndarray), "x must be a jnp.ndarray"
        return jnp.all(x >= self._low) and jnp.all(x <= self._high)

    def __repr__(self) -> str:
        return f"Box(low={self._low}, high={self._high})"


class Tuple(AbstractSpace[tuple[Any, ...]], strict=True):
    """A cartesian product of spaces."""

    spaces: tuple[AbstractSpace, ...]

    def __init__(self, spaces: tuple[AbstractSpace, ...]):
        assert isinstance(spaces, tuple), "spaces must be a tuple"
        assert len(spaces) > 0, "spaces must be non-empty"
        assert all(
            isinstance(space, AbstractSpace) for space in spaces
        ), "spaces must be a tuple of AbstractSpace"

        self.spaces = spaces

    @property
    def shape(self) -> None:
        return None

    def sample(self, key: Key) -> tuple[Any, ...]:
        return tuple(
            space.sample(key)
            for space, key in zip(self.spaces, jr.split(key, len(self.spaces)))
        )

    def contains(self, x: Any) -> Bool[Array, ""]:
        if not isinstance(x, tuple):
            return jnp.array(False)

        if len(x) != len(self.spaces):
            return jnp.array(False)

        return jnp.array(all(space.contains(x_i) for space, x_i in zip(self.spaces, x)))

    def __repr__(self) -> str:
        return f"Tuple({', '.join(repr(space) for space in self.spaces)})"

    def __getitem__(self, index: int) -> AbstractSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)


class Dict(AbstractSpace[dict[str, Any]], strict=True):
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

    def sample(self, key: Key) -> dict[str, Any]:
        return {
            space: self.spaces[space].sample(key)
            for space, key in zip(self.spaces.keys(), jr.split(key, len(self.spaces)))
        }

    def contains(self, x: Any) -> Bool[Array, ""]:
        if not isinstance(x, dict):
            return jnp.array(False)

        if len(x) != len(self.spaces):
            return jnp.array(False)

        return jnp.array(
            all(
                key in self.spaces and self.spaces[key].contains(x[key])
                for key in x.keys()
            )
        )

    def __repr__(self) -> str:
        return f"Dict({', '.join(f'{key}: {repr(space)}' for key, space in self.spaces.items())})"


class MultiDiscrete(AbstractSpace[Int[ArrayLike, " n"]], strict=True):
    """Cartesian product of discrete spaces."""

    ns: Int[Array, " n"]
    starts: Int[Array, " n"]

    def __init__(self, ns: tuple[int, ...], starts: tuple[int, ...] = (0,)):
        assert len(ns) > 0, "ns must be non-empty"
        starts = tuple(starts) if len(starts) > 0 else (0,) * len(ns)
        assert len(ns) == len(starts), "ns and starts must have the same length"
        assert all(n > 0 for n in ns), "all n must be positive"

        self.ns = jnp.asarray(ns)
        self.starts = jnp.asarray(starts)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.ns),)

    def sample(self, key: Key) -> Int[Array, " n"]:
        return jr.randint(
            key, shape=self.shape, minval=self.starts, maxval=self.ns + self.starts
        )

    def contains(self, x: Any) -> Bool[Array, ""]:
        if not isinstance(x, jnp.ndarray):
            return jnp.array(False)

        if x.shape != self.shape:
            return jnp.array(False)

        return jnp.all((self.starts <= x) & (x < self.ns + self.starts), axis=0)

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.ns}, starts={self.starts})"
