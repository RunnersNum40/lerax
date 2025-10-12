from __future__ import annotations

from typing import Callable, NamedTuple

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from lerax.space import Box


class RescaleResult(NamedTuple):
    box: Box
    forward: Callable[[Float[ArrayLike, " ..."]], Float[Array, " ..."]]
    backward: Callable[[Float[ArrayLike, " ..."]], Float[Array, " ..."]]


def rescale_box(
    box: Box, min: Float[ArrayLike, " ..."], max: Float[ArrayLike, " ..."]
) -> RescaleResult:
    """
    Return functions to rescale one box space to another.

    Logic stolen from Gymnasium library

    Returns:
        A rescaled box, a function to map from the original to the rescaled box, and a
        function to map from the rescaled box to the original.
    """

    min = jnp.broadcast_to(min, box.shape)
    max = jnp.broadcast_to(max, box.shape)

    assert jnp.all(min <= max)
    assert jnp.all(box.low <= box.high)

    min_finite = jnp.isfinite(min)
    max_finite = jnp.isfinite(max)
    both_finite = min_finite & max_finite

    gradient = jnp.ones_like(min)
    gradient = gradient.at[both_finite].set(
        (max[both_finite] - min[both_finite])
        / (box.high[both_finite] - box.low[both_finite])
    )

    intercept = jnp.zeros_like(min)
    intercept = intercept.at[max_finite].set(max[max_finite] - box.high[max_finite])
    intercept = intercept.at[min_finite].set(
        min[min_finite] - box.low[min_finite] * gradient[min_finite]
    )

    new_box = Box(low=min, high=max, shape=box.shape)

    def forward(sample: Float[ArrayLike, " ..."]) -> Float[Array, " ..."]:
        sample = jnp.asarray(sample)
        return gradient * sample + intercept

    def backward(sample: Float[ArrayLike, " ..."]) -> Float[Array, " ..."]:
        sample = jnp.asarray(sample)
        return (sample - intercept) / gradient

    return RescaleResult(new_box, forward, backward)
