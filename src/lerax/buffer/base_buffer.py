from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Integer, Key


class AbstractBuffer(eqx.Module):
    """Abstract base class for experience buffers.

    Provides common flattening, batching, and gathering utilities.
    Subclasses must implement `sample` and `shape`.
    """

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

    def batch_indices(
        self,
        batch_size: int,
        *,
        key: Key[Array, ""] | None = None,
    ) -> Integer[Array, "num_batches batch_size"]:
        """
        Return shuffled index groups for lazy minibatch gathering.

        The returned array has shape ``(num_batches, batch_size)`` and can
        be passed as the ``xs`` argument of a scan loop.  Inside the loop
        body, call :meth:`gather` with one row to materialise a single
        minibatch without allocating all batches at once.

        Args:
            batch_size: Number of samples per minibatch.
            key: PRNG key for shuffling. If ``None``, indices are sequential.

        Returns:
            Index array of shape ``(num_batches, batch_size)``.
        """
        total = self.shape[0]
        indices = jnp.arange(total) if key is None else jr.permutation(key, total)
        total_trim = total - (total % batch_size)
        return indices[:total_trim].reshape(-1, batch_size)

    def gather[SelfType: AbstractBuffer](
        self: SelfType,
        indices: Integer[Array, " batch_size"],
    ) -> SelfType:
        """
        Gather a single minibatch by index.

        Args:
            indices: 1-D index array of shape ``(batch_size,)``.

        Returns:
            A buffer containing only the selected samples.
        """
        return jax.tree.map(lambda x: jnp.take(x, indices, axis=0), self)

    def batches[SelfType: AbstractBuffer](
        self: SelfType,
        batch_size: int,
        *,
        key: Key[Array, ""] | None = None,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> SelfType:
        """
        Materialise all minibatches as a single stacked buffer.

        This is convenient but allocates ``(num_batches, batch_size, ...)``
        up front.  For memory-constrained training loops, prefer
        :meth:`batch_indices` with :meth:`gather` inside a scan instead.

        Args:
            batch_size: Number of samples per minibatch.
            key: PRNG key for shuffling. If ``None``, indices are sequential.
            batch_axes: Axes to flatten before batching.

        Returns:
            Buffer with an extra leading ``num_batches`` dimension.
        """
        flat_self = self.flatten_axes(batch_axes)
        indices = flat_self.batch_indices(batch_size, key=key)
        return jax.tree.map(lambda x: jnp.take(x, indices, axis=0), flat_self)

    @abstractmethod
    def sample[SelfType: AbstractBuffer](
        self: SelfType,
        batch_size: int,
        *,
        key: Key[Array, ""],
    ) -> SelfType:
        """Return uniformly sampled batch of data from the buffer."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the buffer."""
