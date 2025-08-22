from __future__ import annotations

import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.model.mlp import MLP


class TestMLP:
    def test_interface(self):
        key = jr.key(0)
        m = MLP(in_size=7, out_size=4, width_size=10, depth=2, key=key)
        x = jnp.zeros(7)
        y1 = m(x)
        y2 = m(x)
        assert isinstance(y1, jnp.ndarray)
        assert y1.shape == (4,)
        assert jnp.array_equal(y1, y2)

    def test_jit(self):
        key = jr.key(0)
        m = MLP(in_size=5, out_size=3, width_size=8, depth=1, key=key)
        x = jr.normal(jr.key(1), (5,))

        @eqx.filter_jit
        def f(mm, xx):
            return mm(xx)

        y = f(m, x)
        assert y.shape == (3,)

    def test_bad_input_shape_raises(self):
        key = jr.key(0)
        m = MLP(in_size=5, out_size=3, width_size=8, depth=1, key=key)
        with pytest.raises(Exception):
            _ = m(jnp.ones((6,)))
