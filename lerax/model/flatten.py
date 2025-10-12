from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array, Float

from .base_model import AbstractModel


class Flatten(AbstractModel[[Float[Array, "..."]], Float[Array, " out_size"]]):
    """Warps the JAX `jnp.ravel` function to flatten an input array."""

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, " out_size"]:
        """Flatten the input array into a 1D array."""
        return jnp.ravel(x)
