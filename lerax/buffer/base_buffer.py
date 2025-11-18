from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Key


class AbstractBuffer(eqx.Module):
    def resolve_axes(
        self,
        batch_axes: tuple[int, ...] | int | None,
    ) -> tuple[int, ...]:
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

        return axes

    def flatten_axes[SelfType: AbstractBuffer](
        self: SelfType,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> SelfType:
        axes = self.resolve_axes(batch_axes)
        num_axes = len(axes)
        target_axes = tuple(range(num_axes))
        max_axis = max(axes) if axes else -1

        def flatten_leaf(x):
            if not isinstance(x, jnp.ndarray):
                return x

            if x.ndim <= max_axis:
                return x

            moved = jnp.moveaxis(x, axes, target_axes)

            leading = 1
            for i in range(num_axes):
                leading *= moved.shape[i]

            return moved.reshape((leading,) + moved.shape[num_axes:])

        return jax.tree.map(flatten_leaf, self)

    @abstractmethod
    def batches[SelfType: AbstractBuffer](
        self: SelfType,
        batch_size: int,
        *,
        key: Key | None = None,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> SelfType:
        """Return an iterator over batches of data from the buffer."""

    @abstractmethod
    def sample[SelfType: AbstractBuffer](
        self: SelfType,
        batch_size: int,
        *,
        key: Key,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> SelfType:
        """Return uniformly sampled batch of data from the buffer."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the buffer."""
