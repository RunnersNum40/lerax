from __future__ import annotations

from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, overload

import equinox as eqx
import jax
import numpy as np
from jax import lax
from jax import numpy as jnp


# TODO: Consider moving this to a standalone Equinox utilities library
class _FilterScan(eqx.Module):

    @property
    def __wrapped__(self):
        return lax.scan

    def __call__(
        self,
        f,
        init,
        xs=None,
        length=None,
        reverse: bool = False,
        unroll: int | bool = 1,
        _split_transpose: bool = False,
    ):
        init_arr, static = eqx.partition(init, eqx.is_array)

        @eqx.filter_jit
        def _f(carry_arr, x):
            carry = eqx.combine(carry_arr, static)
            carry, y = f(carry, x)
            new_carry_arr, new_static = eqx.partition(carry, eqx.is_array)
            assert eqx.tree_equal(
                static, new_static
            ), "Non-array carry of filter_scan must not change."
            return new_carry_arr, y

        carry_arr, ys = lax.scan(
            f=_f,
            init=init_arr,
            xs=xs,
            length=length,
            reverse=reverse,
            unroll=unroll,
            _split_transpose=_split_transpose,
        )
        return eqx.combine(carry_arr, static), ys


filter_scan = eqx.module_update_wrapper(_FilterScan())


def callback_wrapper[**InType](
    func: Callable[InType, Any], ordered: bool = False
) -> Callable[InType, None]:
    """Return a JIT‑safe version of *func*."""

    def _callback(*args: InType.args, **kwargs: InType.kwargs) -> None:
        func(*args, **kwargs)

    @wraps(func)
    def wrapped(*args: InType.args, **kwargs: InType.kwargs) -> None:
        jax.debug.callback(_callback, *args, **kwargs, ordered=ordered)

    return wrapped


def callback_with_numpy_wrapper(
    func: Callable[..., Any], ordered: bool = False
) -> Callable[..., None]:
    """
    Like `debug_wrapper` but converts every jax.Array/`jnp.ndarray` argument
    to a plain `numpy.ndarray` before calling *func*.

    It is impossible with Python's current type system to express the transformation so
    parameter information is lost.
    """

    @partial(callback_wrapper, ordered=ordered)
    @wraps(func)
    def wrapped(*args, **kwargs) -> None:
        args, kwargs = jax.tree.map(
            lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, (args, kwargs)
        )
        func(*args, **kwargs)

    return wrapped


def callback_with_list_wrapper(
    func: Callable[..., Any], ordered: bool = False
) -> Callable[..., None]:
    """
    Like `debug_wrapper` but converts every jax.Array/`jnp.ndarray` argument
    to a plain list before calling *func*.

    It is impossible with Python's current type system to express the transformation so
    parameter information is lost.
    """

    @partial(callback_wrapper, ordered=ordered)
    @wraps(func)
    def wrapped(*args, **kwargs) -> None:
        args, kwargs = jax.tree.map(
            lambda x: (
                np.asarray(x).tolist()
                if isinstance(x, (jnp.ndarray, np.ndarray))
                else x
            ),
            (args, kwargs),
        )
        func(*args, **kwargs)

    return wrapped


print_callback = callback_with_list_wrapper(print, ordered=True)


def unstack_pytree[T](tree: T, *, axis: int = 0) -> tuple[T]:
    """Split a stacked pytree along `axis` into a tuple of pytrees with the same structure."""

    times = jnp.array(jax.tree.leaves(jax.tree.map(lambda x: x.shape[axis], tree)))
    tree = eqx.error_if(
        tree,
        ~times == times[0],
        "All leaves must have the same size along the specified axis.",
    )

    outer_structure = jax.tree.structure(tree)
    unstacked = jax.tree.map(partial(jnp.unstack, axis=axis), tree)
    transposed = jax.tree.transpose(outer_structure, None, unstacked)
    return transposed


class Serializable(eqx.Module):
    @overload
    def serialize(
        self,
        path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    @overload
    def serialize[**PathArgs](
        self,
        path: Callable[PathArgs, str],
        *args: PathArgs.args,
        **kwargs: PathArgs.kwargs,
    ) -> None:
        pass

    @callback_wrapper
    def serialize[**PathArgs](
        self,
        path: str | Path | Callable[PathArgs, str],
        *args: PathArgs.args,
        **kwargs: PathArgs.kwargs,
    ) -> None:
        """
        Serialize the model to the specified path. Works under JIT.
        If a callable is provided as path, it will be called with the provided
        arguments to obtain the path. If a format string is provided as path,
        then it will be formatted with the provided arguments.

        **Aguments:**
            path: The path to serialize to. If a callable is provided, it will
                be called with the provided `args` and `kwargs` to obtain the
                path.
            args: Additional arguments to pass to the path callable or format
                string.
            kwargs: Additional keyword arguments to pass to the path callable
                or format string.
        """
        if callable(path):
            path = path(*args, **kwargs)
        elif isinstance(path, str) and not (args is None and kwargs is None):
            path = path.format(*args, **kwargs)

        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".eqx":
            path = path.with_suffix(".eqx")

        eqx.tree_serialise_leaves(path, self)

    @classmethod
    def deserialize[**Params, ClassType](
        cls: Callable[Params, ClassType],
        path: str | Path,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ClassType:
        """
        Deserialize the model from the specified path.
        Must provide any additional arguments required by the class constructor.

        **Arguments:**
            path: The path to deserialize from.
            args: Additional arguments to pass to the class constructor
            kwargs: Additional keyword arguments to pass to the class constructor

        **Returns:**
            cls: The deserialized model.
        """
        return eqx.tree_deserialise_leaves(
            path, eqx.filter_eval_shape(cls, *args, **kwargs)
        )
