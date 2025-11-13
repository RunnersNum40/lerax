from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Key


class AbstractBuffer(eqx.Module):
    """Base class for buffers."""

    def batches[SelfType: AbstractBuffer](
        self: SelfType,
        batch_size: int,
        *,
        key: Key | None = None,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> SelfType:
        ndim = len(self.shape)

        if batch_axes is None:
            axes = tuple(range(ndim))
        elif isinstance(batch_axes, int):
            axes = (batch_axes,)
        else:
            axes = tuple(batch_axes)

        axes = tuple(a + ndim if a < 0 else a for a in axes)
        if len(set(axes)) != len(axes) or any(a < 0 or a >= ndim for a in axes):
            raise ValueError(f"Invalid batch_axes {batch_axes} for array ndim={ndim}.")

        def flatten_selected(x):
            moved = jnp.moveaxis(x, axes, tuple(range(len(axes))))
            lead = 1
            for i in range(len(axes)):
                lead *= moved.shape[i]

            return moved.reshape((lead,) + moved.shape[len(axes) :])

        flat_self = jax.tree.map(flatten_selected, self)

        total = flat_self.rewards.shape[0]
        indices = jnp.arange(total) if key is None else jr.permutation(key, total)

        if total % batch_size != 0:
            total_trim = total - (total % batch_size)
            indices = indices[:total_trim]

        indices = indices.reshape(-1, batch_size)

        return jax.tree.map(
            lambda x: jnp.take(x, indices, axis=0) if isinstance(x, jnp.ndarray) else x,
            flat_self,
        )

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the buffer."""
