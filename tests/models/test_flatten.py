from __future__ import annotations

import equinox as eqx
import pytest
from jax import numpy as jnp

from oryx.model.flatten import Flatten


class TestFlatten:
    @pytest.mark.parametrize("dims", range(4))
    def test_interface(self, dims: int):
        model = Flatten()
        x = jnp.ones((2,) * dims)
        y = model(x)
        assert isinstance(y, jnp.ndarray)
        assert y.shape == (2**dims,)
        assert jnp.array_equal(y, jnp.ones((2**dims,)))

    def test_jit(self):
        model = Flatten()
        x = jnp.arange(6.0).reshape(2, 3)

        @eqx.filter_jit
        def f(m, a):
            return m(a)

        y = f(model, x)
        assert jnp.array_equal(y, x.ravel())
