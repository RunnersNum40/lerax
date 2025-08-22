from __future__ import annotations

import diffrax
import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.model.node.node import MLPNeuralODE


class TestMLPNeuralODE:
    @pytest.mark.parametrize("time_in_input", [True, False])
    def test_interface(self, time_in_input: bool):
        key = jr.key(0)
        model = MLPNeuralODE(
            in_size=3,
            out_size=2,
            latent_size=4,
            width_size=6,
            depth=1,
            key=key,
            time_in_input=time_in_input,
        )
        ts = jnp.linspace(0.0, 1.0, num=5)
        z0 = jnp.zeros((4,))
        zs = model.solve(ts, z0)
        assert zs.shape == (5, 4)
        x0 = jnp.ones((3,))
        ys = model(ts, x0)
        assert ys.shape == (5, 2)

    def test_jit(self):
        key = jr.key(0)
        model = MLPNeuralODE(
            in_size=2,
            out_size=3,
            latent_size=2,
            width_size=5,
            depth=1,
            key=key,
        )
        ts = jnp.array([0.0, 0.2, 0.5, 1.0])
        x0 = jnp.ones((2,))

        @eqx.filter_jit
        def f(m, t, x):
            return m(t, x)

        ys = f(model, ts, x0)
        assert ys.shape == (4, 3)

    def test_solvers(self):
        key = jr.key(0)
        ts = jnp.array([0.0, 0.2, 0.5, 1.0])
        z0 = jnp.zeros((2,))
        m1 = MLPNeuralODE(2, 3, 2, 5, 1, key=key)  # default Tsit5
        m2 = MLPNeuralODE(2, 3, 2, 5, 1, key=key, solver=diffrax.Euler)
        zs1 = m1.solve(ts, z0)
        zs2 = m2.solve(ts, z0)
        assert zs1.shape == zs2.shape == (len(ts), 2)
