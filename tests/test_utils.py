import equinox as eqx
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.utils import clone_state, filter_scan


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
