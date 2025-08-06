import threading
import time

import equinox as eqx
import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.utils import clone_state, debug_with_numpy_wrapper, debug_wrapper, filter_scan


class DummyModule(eqx.Module):
    state_index: eqx.nn.StateIndex[int]

    def __init__(self):
        self.state_index = eqx.nn.StateIndex(0)


def test_clone_state_independence():
    module, state = eqx.nn.make_with_state(DummyModule)()
    clone = clone_state(state)

    assert state is not clone
    assert state.tree_flatten() == clone.tree_flatten()

    clone_modified = clone.set(module.state_index, 42)
    assert state.get(module.state_index) == 0
    assert clone_modified.get(module.state_index) == 42


def test_cannot_reuse_state():
    module, state = eqx.nn.make_with_state(DummyModule)()

    state.set(module.state_index, 1)

    with pytest.raises(ValueError):
        state.get(module.state_index)


def test_filter_scan_accumulates_and_keeps_static():
    key = jr.key(0)
    mlp = eqx.nn.MLP("scalar", "scalar", 0, 0, key=key)

    def step(carry, x):
        mlp = carry
        val = mlp(x)
        carry = mlp
        return carry, [val]

    xs = jnp.arange(4)
    with pytest.raises(TypeError):
        lax.scan(step, mlp, xs)

    filter_scan(step, mlp, xs)


def _compute(x):
    _compute.thread_id = threading.get_ident()
    _compute.payload = x


@pytest.mark.parametrize("thread_flag", [False, True])
def test_debug_wrapper_threading(thread_flag):
    """
    Ensure the callback fires and – when requested – in a different thread.
    """
    wrapped = debug_wrapper(_compute, thread=thread_flag)

    @jax.jit
    def f(x):
        wrapped(x)
        return x + 1

    main_thread = threading.get_ident()
    result = f(3.14)
    time.sleep(0.05)

    assert float(result) == pytest.approx(4.14)
    assert hasattr(_compute, "thread_id"), "callback never executed"

    if thread_flag:
        assert _compute.thread_id != main_thread, "expected a different thread"
    else:
        assert _compute.thread_id == main_thread, "should run on main thread"


def _array_collector(x):
    _array_collector.received_type = type(x)


def test_debug_with_numpy_wrapper_converts_arrays():
    wrapped = debug_with_numpy_wrapper(_array_collector)

    @jax.jit
    def f(x):
        wrapped(x)
        return x

    f(jnp.ones((2, 2)))
    time.sleep(0.01)

    assert _array_collector.received_type is np.ndarray
