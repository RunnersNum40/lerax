from typing import cast

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Num, PyTree

from .base_space import (
    AbstractSpace,
    Box,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    OneOf,
)
from .base_space import (
    Dict as DictSpace,
)
from .base_space import (
    Tuple as TupleSpace,
)


def flatten(
    space: AbstractSpace, sample: PyTree[Num[ArrayLike, "..."]]
) -> Num[Array, " size"]:
    """
    Convert sample from an arbitrary Oryx space into a 1-D array.
    """
    if isinstance(space, Box):
        return jnp.asarray(sample).ravel()

    if isinstance(space, Discrete):
        return jnp.asarray([sample])

    if isinstance(space, MultiBinary):
        return jnp.asarray(sample).ravel()

    if isinstance(space, MultiDiscrete):
        return jnp.asarray(sample).ravel()

    if isinstance(space, TupleSpace):
        parts = [
            flatten(subspace, sub_sample)
            for subspace, sub_sample in zip(space.spaces, cast(tuple, sample))
        ]
        return jnp.concatenate(parts)

    if isinstance(space, DictSpace):
        parts = [
            flatten(space.spaces[key], sample[key])
            for key in sorted(space.spaces.keys())
        ]
        return jnp.concatenate(parts)

    if isinstance(space, OneOf):
        # TODO: Implement OneOf flattening
        raise NotImplementedError("Flattening is currenty not implemented for OneOf")

    raise NotImplementedError(f"Flattening not implemented for space {type(space)}.")
