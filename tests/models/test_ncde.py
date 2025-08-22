from __future__ import annotations

import diffrax
import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.model.ncde.ncde import MLPNeuralCDE
from oryx.model.ncde.term import AbstractNCDETerm, MLPNCDETerm


class TestNCDETerm:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractNCDETerm()  # pyright: ignore

    def test_missing_call_keeps_abstract(self):
        class NoCall(AbstractNCDETerm):  # type: ignore
            pass

        with pytest.raises(TypeError):
            NoCall()  # pyright: ignore

    @pytest.mark.parametrize("add_time", [True, False])
    def test_mlp_term_output_shape(self, add_time: bool):
        key = jr.key(0)
        term = MLPNCDETerm(
            input_size=3, data_size=5, width_size=8, depth=2, key=key, add_time=add_time
        )
        t = 0.25
        z = jnp.ones((5,))
        out = term(t, z, None)
        assert out.shape == (5, 3)


class TestMLPNeuralCDE:
    def test_interface(self):
        key = jr.key(0)
        in_size, out_size, latent_size, state_size = 4, 2, 6, 4
        model, state = eqx.nn.make_with_state(MLPNeuralCDE)(
            in_size=in_size,
            out_size=out_size,
            latent_size=latent_size,
            width_size=7,
            depth=1,
            key=key,
            state_size=state_size,
            inference=False,
        )
        state, y = model(state, jnp.zeros(()), jnp.zeros((in_size,)))
        state, y = model(state, jnp.ones(()), jnp.ones((in_size,)))
        assert y.shape == (out_size,)
        assert model.t1(state).shape == ()
        assert model.t1(state) == 1.0
        assert model.ts(state).shape == (state_size,)
        assert model.x1(state).shape == (in_size,)
        assert model.xs(state).shape == (state_size, in_size)
        assert model.z1(state).shape == (latent_size,)
        assert model.zs(state).shape == (state_size, latent_size)

    def test_jit(self):
        key = jr.key(0)
        model, state = eqx.nn.make_with_state(MLPNeuralCDE)(
            in_size=3,
            out_size=2,
            latent_size=4,
            width_size=5,
            depth=1,
            key=key,
            state_size=3,
            inference=True,
        )
        t = jnp.asarray(0.0)
        x = jnp.zeros((3,))

        @eqx.filter_jit
        def f(m, st, tt, xx):
            st, y = m(st, tt, xx)
            return st, y

        state, y = f(model, state, t, x)
        assert y.shape == (2,)

    def test_initial_insert_at_zero(self):
        key = jr.key(0)
        model, state = eqx.nn.make_with_state(MLPNeuralCDE)(
            in_size=2,
            out_size=1,
            latent_size=3,
            width_size=4,
            depth=1,
            key=key,
            state_size=4,
        )
        t, x = jnp.asarray(0.0), jnp.zeros((2,))
        state, _ = model(state, t, x)
        ts = model.ts(state)
        assert ts[0] == 0.0 and not jnp.isnan(ts[0]) and jnp.isnan(ts[1:]).all()

    @pytest.mark.parametrize("time_in_input", [True, False])
    def test_z0_and_coeffs_shapes(self, time_in_input: bool):
        key = jr.key(0)
        model = MLPNeuralCDE(
            in_size=3,
            out_size=2,
            latent_size=5,
            width_size=4,
            depth=1,
            key=key,
            time_in_input=time_in_input,
        )
        t0 = jnp.array(0.0)
        x0 = jnp.ones((3,))
        z0 = model.z0(t0, x0)
        assert z0.shape == (5,)

        ts = jnp.linspace(0.0, 1.0, 6)
        xs = jnp.zeros((6, 3))
        coeffs = model.coeffs(ts, xs)
        expected_channels = 3 + int(time_in_input)
        assert len(coeffs) == 4
        for c in coeffs:
            assert c.shape == (len(ts) - 1, expected_channels)

    @pytest.mark.parametrize("inference", [True, False])
    def test_solve_output_shape(self, inference: bool):
        key = jr.key(0)
        model = MLPNeuralCDE(
            in_size=2,
            out_size=1,
            latent_size=4,
            width_size=4,
            depth=1,
            key=key,
            solver=diffrax.Euler,
            inference=inference,
        )
        ts = jnp.linspace(0.0, 1.0, 5)
        xs = jnp.zeros((5, 2))
        coeffs = model.coeffs(ts, xs)
        z0 = model.z0(ts[0], xs[0])
        zs = model.solve(ts, z0, coeffs)
        assert zs.shape == (len(ts), 4)

    @pytest.mark.parametrize("inference", [True, False])
    def test_state_rollover(self, inference: bool):
        key = jr.key(0)
        model, state = eqx.nn.make_with_state(MLPNeuralCDE)(
            in_size=2,
            out_size=1,
            latent_size=3,
            width_size=4,
            depth=1,
            key=key,
            state_size=3,
            solver=diffrax.Euler,
            inference=inference,
        )
        for t in jnp.arange(4.0):
            x = jnp.full((2,), t)
            state, y = model(state, t, x)
            assert y.shape == (1,)

        ts = model.ts(state)
        assert not jnp.isnan(ts).any()
        assert jnp.allclose(ts, jnp.array([1.0, 2.0, 3.0]))
        assert model.t1(state) == 3.0
