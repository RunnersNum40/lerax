from collections.abc import Callable

import equinox as eqx
import jax


def clone_state(state: eqx.nn.State) -> eqx.nn.State:
    """
    Clone an Equinox state.

    Equinox does not allow reuse of states. Cloning in this way bypasses this restriction.
    """
    leaves, treedef = jax.tree.flatten(state)
    state_clone = jax.tree.unflatten(treedef, leaves)
    return state_clone


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

    carry_arr, y = jax.lax.scan(
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
