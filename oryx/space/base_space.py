from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key


def try_cast(x: Any) -> Array | None:
    try:
        return jnp.asarray(x)
    except TypeError:
        return None


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
    def contains(self, x: Any) -> Bool[ArrayLike, ""]:
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
    def flatten_sample(self, sample: SampleType) -> Float[Array, "..."]:
        """Flattens a sample from the space into a 1-D array."""

    @property
    @abstractmethod
    def flat_dim(self) -> Int[ArrayLike, ""]:
        """Returns the dimension of the flattened sample."""


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

    def contains(self, x: Any) -> Bool[ArrayLike, ""]:
        x = try_cast(x)
        if x is None:
            return False

        if x.ndim != 0:
            return False
        x = x.squeeze()

        if jnp.logical_not(jnp.array_equal(x, jnp.floor(x))):
            return False

        return bool(self.start <= x < self._n + self.start)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Discrete):
            return False
        return bool(self._n == other._n) and bool(self.start == other.start)

    def __repr__(self) -> str:
        return f"Discrete({self._n}, start={self.start})"

    def __hash__(self) -> int:
        return hash((int(self._n), int(self.start)))

    def flatten_sample(self, sample: Int[Array, ""]) -> Float[Array, ""]:
        return jnp.asarray(sample, dtype=float)

    @property
    def flat_dim(self) -> Int[ArrayLike, ""]:
        return jnp.array(1, dtype=int)

    @property
    def dtype(self) -> jnp.dtype:
        return self._n.dtype


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

    def contains(self, x: Any) -> Bool[ArrayLike, ""]:
        x = try_cast(x)
        if x is None:
            return False

        if x.shape != self._shape:
            return False

        return jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))

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

    def flatten_sample(self, sample: Float[Array, " ..."]) -> Float[Array, " size"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_dim(self) -> Int[ArrayLike, ""]:
        return jnp.prod(jnp.asarray(self._shape)).astype(int)

    @property
    def dtype(self) -> jnp.dtype:
        return self.low.dtype


class Tuple(AbstractSpace[tuple[Any, ...]]):
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

    def canonical(self) -> tuple[Any, ...]:
        return tuple(space.canonical() for space in self.spaces)

    def sample(self, key: Key) -> tuple[Any, ...]:
        return tuple(
            space.sample(key)
            for space, key in zip(self.spaces, jr.split(key, len(self.spaces)))
        )

    def contains(self, x: Any) -> Bool[ArrayLike, ""]:
        if not isinstance(x, tuple):
            return False

        if len(x) != len(self.spaces):
            return False

        return jnp.array(
            space.contains(x_i) for space, x_i in zip(self.spaces, x)
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
    def flat_dim(self) -> Int[ArrayLike, ""]:
        return jnp.array(space.flat_dim for space in self.spaces).sum().astype(int)

    def __getitem__(self, index: int) -> AbstractSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)


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

    def contains(self, x: Any) -> Bool[ArrayLike, ""]:
        if not isinstance(x, dict):
            return False

        if len(x) != len(self.spaces):
            return False

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
    def flat_dim(self) -> Int[ArrayLike, ""]:
        return (
            jnp.array(space.flat_dim for space in self.spaces.values())
            .sum()
            .astype(int)
        )

    def __getitem__(self, index: str) -> AbstractSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)


class MultiDiscrete(AbstractSpace[Int[ArrayLike, " n"]]):
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

    def contains(self, x: Any) -> Bool[ArrayLike, ""]:
        x = try_cast(x)
        if x is None:
            return False

        if x.shape != self.shape:
            return False

        if jnp.logical_not(jnp.array_equal(x, jnp.floor(x))):
            return False

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
    def flat_dim(self) -> Int[ArrayLike, ""]:
        return jnp.array(len(self.ns), dtype=int)


class MultiBinary(AbstractSpace[Bool[Array, " n"]]):
    """A space of binary values."""

    n: int

    def __init__(self, n: int):
        assert n > 0, "n must be positive"
        self.n = n

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,)

    def canonical(self) -> Bool[Array, " n"]:
        return jnp.zeros(self.shape, dtype=bool)

    def sample(self, key: Key) -> Bool[Array, " n"]:
        return jr.bernoulli(key, shape=self.shape)

    def contains(self, x: Any) -> Bool[ArrayLike, ""]:
        x = try_cast(x)
        if x is None:
            return False

        if x.shape != self.shape:
            return False

        return jnp.all((x == 0) | (x == 1), axis=0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiBinary):
            return False

        return bool(self.n == other.n)

    def __repr__(self) -> str:
        return f"MultiBinary({self.n})"

    def __hash__(self) -> int:
        return hash(self.n)

    def flatten_sample(self, sample: Bool[ArrayLike, " n"]) -> Float[Array, " size"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_dim(self) -> Int[ArrayLike, ""]:
        return jnp.array(self.n, dtype=int)
