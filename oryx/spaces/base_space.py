from abc import abstractmethod
from typing import Any

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key


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
    def sample(self, key: Key) -> SampleType:
        """Returns a random sample from the space."""

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Returns True if the input is in the space, False otherwise."""

    def __contains__(self, x: Any) -> bool:
        return self.contains(x)

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Checks if two spaces are equal based on their properties."""

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the space."""


class Discrete(AbstractSpace[Int[Array, ""]]):
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

    def contains(self, x: Any) -> bool:
        if not isinstance(x, (int, jnp.ndarray, np.ndarray)):
            return False

        return bool(self.start <= x < self._n + self.start)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Discrete):
            return False
        return bool(self._n == other._n) and bool(self.start == other.start)

    def __repr__(self) -> str:
        return f"Discrete({self._n}, start={self.start})"


class Box(AbstractSpace[Float[Array, " ..."]]):
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
        shape: tuple[int, ...] | None = None,
    ):
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        if shape is None:
            low, high = jnp.broadcast_arrays(low, high)
            shape = low.shape
            # TODO: Add warning if both shapes change

        assert low.shape == high.shape, "Box low and high must have the same shape"

        self._shape = shape
        self._low = jnp.broadcast_to(low, shape)
        self._high = jnp.broadcast_to(high, shape)

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

    def contains(self, x: Any) -> bool:
        if not isinstance(x, (jnp.ndarray, np.ndarray)):
            return False

        return bool(jnp.all(x >= self._low)) and bool(jnp.all(x <= self._high))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return False
        return bool(jnp.array_equal(self._low, other._low)) and bool(
            jnp.array_equal(self._high, other._high)
        )

    def __repr__(self) -> str:
        return f"Box(low={self._low}, high={self._high})"


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

    def sample(self, key: Key) -> tuple[Any, ...]:
        return tuple(
            space.sample(key)
            for space, key in zip(self.spaces, jr.split(key, len(self.spaces)))
        )

    def contains(self, x: Any) -> bool:
        if not isinstance(x, tuple):
            return False

        if len(x) != len(self.spaces):
            return False

        return all(space.contains(x_i) for space, x_i in zip(self.spaces, x))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tuple):
            return False
        return all(
            space == other_space
            for space, other_space in zip(self.spaces, other.spaces)
        )

    def __repr__(self) -> str:
        return f"Tuple({', '.join(repr(space) for space in self.spaces)})"

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

    def sample(self, key: Key) -> dict[str, Any]:
        return {
            space_key: self.spaces[space_key].sample(rng_key)
            for space_key, rng_key in zip(
                self.spaces.keys(), jr.split(key, len(self.spaces))
            )
        }

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False

        if len(x) != len(self.spaces):
            return False

        return all(
            key in self.spaces and self.spaces[key].contains(x[key]) for key in x.keys()
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dict):
            return False
        return all(
            key in other.spaces and self.spaces[key] == other.spaces[key]
            for key in self.spaces.keys()
        )

    def __repr__(self) -> str:
        return f"Dict({', '.join(f'{key}: {repr(space)}' for key, space in self.spaces.items())})"

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

        self.ns = jnp.asarray(ns)
        self.starts = jnp.asarray(starts)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.ns),)

    def sample(self, key: Key) -> Int[Array, " n"]:
        return jr.randint(
            key, shape=self.shape, minval=self.starts, maxval=self.ns + self.starts
        )

    def contains(self, x: Any) -> bool:
        if not isinstance(x, jnp.ndarray):
            return False

        if x.shape != self.shape:
            return False

        return bool(jnp.all((self.starts <= x) & (x < self.ns + self.starts), axis=0))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDiscrete):
            return False
        return bool(jnp.array_equal(self.ns, other.ns)) and bool(
            jnp.array_equal(self.starts, other.starts)
        )

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.ns}, starts={self.starts})"


class MultiBinary(AbstractSpace[Bool[Array, " n"]]):
    """A space of binary values."""

    n: int

    def __init__(self, n: int):
        assert n > 0, "n must be positive"
        self.n = n

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,)

    def sample(self, key: Key) -> Bool[Array, " n"]:
        return jr.bernoulli(key, shape=self.shape)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, jnp.ndarray):
            return False

        if x.shape != self.shape:
            return False

        return bool(jnp.all((x == 0) | (x == 1), axis=0))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiBinary):
            return False
        return bool(self.n == other.n)

    def __repr__(self) -> str:
        return f"MultiBinary({self.n})"


class OneOf(AbstractSpace):
    """An exclusive tuple of multiple spaces."""

    spaces: tuple[AbstractSpace, ...]

    def __init__(self, spaces: tuple[AbstractSpace, ...]):
        assert len(spaces) > 0, "spaces must be non-empty"

        self.spaces = spaces

    @property
    def shape(self) -> None:
        return None

    def sample(self, key: Key):
        space_key, sample_key = jr.split(key, 2)
        subspace_idx = jr.randint(
            space_key, shape=(), minval=0, maxval=len(self.spaces)
        )
        subspace = self.spaces[subspace_idx]
        return subspace.sample(sample_key)

    def contains(self, x: Any) -> bool:
        return any(space.contains(x) for space in self.spaces)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OneOf):
            return False
        spaces = list(self.spaces)
        try:
            for other_space in other.spaces:
                spaces.remove(other_space)
        except ValueError:
            return False
        return not spaces

    def __repr__(self) -> str:
        return f"OneOf({', '.join(repr(space) for space in self.spaces)})"

    def __getitem__(self, index: int) -> AbstractSpace:
        return self.spaces[index]

    def __len__(self) -> int:
        return len(self.spaces)
