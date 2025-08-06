import threading
from functools import partial, wraps
from typing import Any, Callable

import equinox as eqx
import jax
import numpy as np
from jax import lax
from jax import numpy as jnp


def clone_state(state: eqx.nn.State) -> eqx.nn.State:
    """
    Clone an Equinox state.

    Equinox does not allow reuse of states. Cloning in this way bypasses this restriction.
    """
    leaves, treedef = jax.tree.flatten(state)
    state_clone = jax.tree.unflatten(treedef, leaves)
    return state_clone


@wraps(lax.scan)
def filter_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Y]:
    """
    Allows to use `jax.lax.scan` with a non array carry.
    """
    init_arr, static = eqx.partition(init, eqx.is_array)

    def _f(carry_arr, x):
        carry = eqx.combine(carry_arr, static)
        carry, y = f(carry, x)
        carry_arr, _static = eqx.partition(carry, eqx.is_array)

        eqx.error_if(
            carry_arr,
            _static != static,
            "Non-array carry of filter_scan must not change.",
        )
        return carry_arr, y

    carry_arr, y = lax.scan(
        f=_f,
        init=init_arr,
        xs=xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )

    carry = eqx.combine(carry_arr, static)
    return carry, y


def debug_wrapper[**InType](
    func: Callable[InType, Any], ordered: bool = False, thread: bool = False
) -> Callable[InType, None]:
    """
    Return a JIT‑safe version of *func*.

    :param func: The function to wrap.
    :param ordered: If True, the callback will be executed in the order of the arguments
    :param thread: If True, the callback will be executed in a separate thread.
    """
    if ordered and thread:
        # TODO: Add a warning or error here
        pass

    def _callback(*args: InType.args, **kwargs: InType.kwargs) -> None:
        if thread:
            threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
        else:
            func(*args, **kwargs)

    @wraps(func)
    def wrapped(*args: InType.args, **kwargs: InType.kwargs) -> None:
        jax.debug.callback(_callback, *args, **kwargs, ordered=ordered)

    return wrapped


def debug_with_numpy_wrapper[**InType](
    func: Callable[InType, Any], ordered: bool = False, thread: bool = False
) -> Callable[InType, None]:
    """
    Like `debug_wrapper` but converts every jax.Array/`jnp.ndarray` argument
    to a plain numpy.ndarray` before calling *func*.
    """

    @partial(debug_wrapper, ordered=ordered, thread=thread)
    @wraps(func)
    def wrapped(*args: InType.args, **kwargs: InType.kwargs) -> None:
        args, kwargs = jax.tree_util.tree_map(
            lambda x: np.asarray(x) if isinstance(x, (jax.Array, jnp.ndarray)) else x,
            (args, kwargs),
        )
        func(*args, **kwargs)

    return wrapped
