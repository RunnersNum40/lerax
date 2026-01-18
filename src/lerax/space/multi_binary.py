from __future__ import annotations

import operator
from functools import reduce
from typing import Any

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from .base_space import AbstractSpace
from .utils import try_cast


class MultiBinary(AbstractSpace[Bool[Array, " n"], None]):
    """
    A space of binary values.

    All dimensions are independent and can take values 0 or 1.

    Attributes:
        n: The shape of the multi-binary space.

    Args:
        n: The shape of the multi-binary space. If an integer is provided,
            then the shape will be (n,).
    """

    n: tuple[int, ...]

    def __init__(self, n: int | tuple[int, ...]):
        if isinstance(n, int):
            assert n > 0, "n must be positive"
            self.n = (n,)
        else:
            assert all(
                isinstance(dim, int) and dim > 0 for dim in n
            ), "all dimensions in n must be positive integers"
            self.n = n

    @property
    def shape(self) -> tuple[int, ...]:
        return self.n

    def canonical(self) -> Bool[Array, " n"]:
        return jnp.zeros(self.shape, dtype=bool)

    def sample(self, *, key: Key, mask: None = None) -> Bool[Array, " n"]:
        return jr.bernoulli(key, shape=self.shape)

    def contains(self, x: Any) -> Bool[Array, ""]:
        x = try_cast(x)
        if x is None:
            return jnp.array(False)

        if x.shape != self.shape:
            return jnp.array(False)

        return jnp.all((x == 0) | (x == 1), axis=0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiBinary):
            return False

        return self.n == other.n

    def __repr__(self) -> str:
        return f"MultiBinary({self.n})"

    def __hash__(self) -> int:
        return hash(self.n)

    def flatten_sample(self, sample: Bool[ArrayLike, " n"]) -> Float[Array, " n"]:
        return jnp.asarray(sample, dtype=float).ravel()

    @property
    def flat_size(self) -> int:
        return reduce(operator.mul, self.n)
