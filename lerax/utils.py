from __future__ import annotations

from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import numpy as np
from jax import lax
from jax import numpy as jnp


def filter_scan(
    f,
    init,
    xs=None,
    length=None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
):
    """
    An easier to use version of `lax.scan`. All JAX and Numpy arrays are
    traced, and only non-array parts of the carry are static.

    Args:
        f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
        that ``f`` accepts two arguments where the first is a value of the loop
        carry and the second is a slice of ``xs`` along its leading axis, and that
        ``f`` returns a pair where the first element represents a new value for
        the loop carry and the second represents a slice of the output.
        init: an initial loop carry value of type ``c``, which can be a scalar,
        array, or any pytree (nested Python tuple/list/dict) thereof, representing
        the initial loop carry value. This value must have the same structure as
        the first element of the pair returned by ``f``.
        xs: the value of type ``[a]`` over which to scan along the leading axis,
        where ``[a]`` can be an array or any pytree (nested Python
        tuple/list/dict) thereof with consistent leading axis sizes.
        length: optional integer specifying the number of loop iterations, which
        must agree with the sizes of leading axes of the arrays in ``xs`` (but can
        be used to perform scans where no input ``xs`` are needed).
        reverse: optional boolean specifying whether to run the scan iteration
        forward (the default) or in reverse, equivalent to reversing the leading
        axes of the arrays in both ``xs`` and in ``ys``.
        unroll: optional non-negative int or bool specifying, in the underlying
        operation of the scan primitive, how many scan iterations to unroll within
        a single iteration of a loop. If an integer is provided, it determines how
        many unrolled loop iterations to run within a single rolled iteration of
        the loop. `unroll=0` unrolls the entire loop.
        If a boolean is provided, it will determine if the loop is
        completely unrolled (i.e. `unroll=True`) or left completely rolled (i.e.
        `unroll=False`).
        _split_transpose: experimental optional bool specifying whether to further
        split the transpose into a scan (computing activation gradients), and a
        map (computing gradients corresponding to the array arguments). Enabling
        this may increase memory requirements, and so is an experimental feature
        that may evolve or even be rolled back.

    Returns:
        A pair of type ``(c, [b])`` where the first element represents the final
        loop carry value and the second element represents the stacked outputs of
        the second output of ``f`` when scanned over the leading axis of the inputs.
    """
    init_arr, static = eqx.partition(init, eqx.is_array)

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


def callback_wrapper[**InType](
    func: Callable[InType, Any], ordered: bool = False
) -> Callable[InType, None]:
    """
    Return a JIT‑safe version of *func*.

    Wraps *func* in a `jax.debug.callback` so that it can be used inside JIT‑compiled
    code.

    Args:
        func: The callback function to wrap.
        ordered: Whether to enforce ordered execution of callbacks.

    Returns:
        A wrapped version of *func* that is JIT-safe.
    """

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

    It is impossible with Python's current type system to express the
    transformation so parameter information is lost.

    Args:
        func: The callback function to wrap.
        ordered: Whether to enforce ordered execution of callbacks.

    Returns:
        A wrapped version of *func* that converts array arguments to numpy
        arrays and is JIT-safe.
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

    It is impossible with Python's current type system to express the
    transformation so parameter information is lost.

    Args:
        func: The callback function to wrap.
        ordered: Whether to enforce ordered execution of callbacks.

    Returns:
        A wrapped version of *func* that converts array arguments to lists and
        is JIT-safe.
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


def unstack_pytree[T](tree: T, *, axis: int = 0) -> Sequence[T]:
    """
    Split a stacked pytree along `axis` into a tuple of pytrees with the same
    structure.

    Args:
        tree: A pytree with array leaves stacked along `axis`.
        axis: The axis along which to unstack the arrays.

    Returns:
        A sequence of pytrees with the same structure, each corresponding to one
        slice along `axis`.
    """

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
    @callback_wrapper
    def serialize(
        self,
        path: str | Path,
        no_suffix: bool = False,
    ) -> None:
        """
        Serialize the model to the specified path.

        Args:
            path: The path to serialize to.
            no_suffix: If True, do not append the ".eqx" suffix
        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".eqx" and not no_suffix:
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

        Args:
            path: The path to deserialize from.
            *args: Additional arguments to pass to the class constructor
            **kwargs: Additional keyword arguments to pass to the class constructor

        Returns:
            The deserialized model.
        """
        return eqx.tree_deserialise_leaves(
            path, eqx.filter_eval_shape(cls, *args, **kwargs)
        )
