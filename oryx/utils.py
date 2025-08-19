from __future__ import annotations

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


@eqx.filter_jit
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
    init_arr, static = eqx.partition(init, eqx.is_array)

    def _f(carry_arr, x):
        carry = eqx.combine(carry_arr, static)
        carry, y = f(carry, x)
        carry_arr, _static = eqx.partition(carry, eqx.is_array)

        # Assert will be omitted from the compiled code
        # I tried using `eqx.error(carry_arr, cond)` if but it breaks if the carry includes a key
        assert eqx.tree_equal(
            static, _static
        ), "Non-array carry of filter_scan must not change."
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


def debug_with_numpy_wrapper(
    func: Callable[..., Any], ordered: bool = False, thread: bool = False
) -> Callable[..., None]:
    """
    Like `debug_wrapper` but converts every jax.Array/`jnp.ndarray` argument
    to a plain numpy.ndarray` before calling *func*.

    It is impossible with Python's current type system to express the transformation so
    parameter information is lost.
    """

    @partial(debug_wrapper, ordered=ordered, thread=thread)
    @wraps(func)
    def wrapped(*args, **kwargs) -> None:
        args, kwargs = jax.tree.map(
            lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, (args, kwargs)
        )
        func(*args, **kwargs)

    return wrapped
