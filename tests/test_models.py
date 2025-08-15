from __future__ import annotations

import diffrax
import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.models.base_model import (
    AbstractModel,
    AbstractStatefulModel,
)
from oryx.models.flatten import Flatten
from oryx.models.mlp import MLP
from oryx.models.ncde.ncde import MLPNeuralCDE
from oryx.models.ncde.term import AbstractNCDETerm, MLPNCDETerm
from oryx.models.node.node import MLPNeuralODE

# TODO: Add tests for stochastic models


def test_cannot_instantiate_abstractmodel_directly():
    with pytest.raises(TypeError):
        AbstractModel()  # pyright: ignore


def test_cannot_instantiate_abstractstatemodel_directly():
    with pytest.raises(TypeError):
        AbstractStatefulModel()  # pyright: ignore


def test_subclass_without_call_fails_abstractmodel():
    class NoCallModel(AbstractModel):
        pass

    with pytest.raises(TypeError):
        NoCallModel()  # pyright: ignore


def test_subclass_without_call_fails_statefulmodel():
    class NoCallStateful(AbstractStatefulModel):
        state_index: eqx.nn.StateIndex[None]

    with pytest.raises(TypeError):
        NoCallStateful()  # pyright: ignore


def test_instantiate_stateful():
    class StatefulModel(AbstractStatefulModel):
        state_index: eqx.nn.StateIndex[None]

        def __init__(self):
            self.state_index = eqx.nn.StateIndex(None)

        def __call__(self, state):
            return state, None

        def reset(self, state):
            return state

    model, state = eqx.nn.make_with_state(StatefulModel)()
    state, _ = model(state)


def test_mlpmodel_forward_shape_and_determinism():
    key = jr.key(0)
    in_size, out_size = 7, 4
    m = MLP(in_size=in_size, out_size=out_size, width_size=10, depth=2, key=key)
    x = jnp.zeros(in_size)
    y1 = m(x)
    y2 = m(x)
    assert isinstance(y1, jnp.ndarray)
    assert y1.shape == (out_size,)
    assert jnp.array_equal(y1, y2)


def test_mlpmodel_wrong_input_shape_raises():
    key = jr.key(1)
    m = MLP(in_size=5, out_size=3, width_size=8, depth=1, key=key)
    x_bad = jnp.ones((6,))
    with pytest.raises(Exception):
        _ = m(x_bad)


@pytest.mark.parametrize("time_in_input", [True, False])
def test_mlpneuralode_solve_and_call(time_in_input: bool):
    key = jr.key(4)
    in_size, out_size, latent_size = 3, 2, 4
    model = MLPNeuralODE(
        in_size=in_size,
        out_size=out_size,
        latent_size=latent_size,
        width_size=6,
        depth=1,
        key=key,
        time_in_input=time_in_input,
    )
    ts = jnp.linspace(0.0, 1.0, num=5)
    z0 = jnp.zeros((latent_size,))
    zs = model.solve(ts, z0)
    assert isinstance(zs, jnp.ndarray)
    assert zs.shape == (5, latent_size)
    x0 = jnp.ones((in_size,))
    ys = model(ts, x0)
    assert isinstance(ys, jnp.ndarray)
    assert ys.shape == (5, out_size)


def test_mlpneuralode_with_different_solvers():
    key = jr.key(5)
    in_size, out_size, latent_size = 2, 3, 2
    ts = jnp.array([0.0, 0.2, 0.5, 1.0])
    z0 = jnp.zeros((latent_size,))
    # default solver (Tsit5)
    model_default = MLPNeuralODE(
        in_size=in_size,
        out_size=out_size,
        latent_size=latent_size,
        width_size=5,
        depth=1,
        key=key,
    )
    zs1 = model_default.solve(ts, z0)
    # Euler solver
    model_euler = MLPNeuralODE(
        in_size=in_size,
        out_size=out_size,
        latent_size=latent_size,
        width_size=5,
        depth=1,
        key=key,
        solver=diffrax.Euler,
    )
    zs2 = model_euler.solve(ts, z0)
    assert zs1.shape == zs2.shape == (len(ts), latent_size)


def test_cannot_instantiate_abstractncdeterm_directly():
    with pytest.raises(TypeError):
        AbstractNCDETerm()  # pyright: ignore


def test_subclass_without_call_fails_ncdeterm():
    class NoCallNCDETerm(AbstractNCDETerm):
        pass

    with pytest.raises(TypeError):
        NoCallNCDETerm()  # pyright: ignore


@pytest.mark.parametrize("add_time", [True, False])
def test_mlpncdeterm_output_shape(add_time: bool):
    key = jr.key(6)
    input_size, data_size = 3, 5
    term = MLPNCDETerm(
        input_size=input_size,
        data_size=data_size,
        width_size=8,
        depth=2,
        key=key,
        add_time=add_time,
    )
    t = 0.25
    z = jnp.ones((data_size,))
    out = term(t, z, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (data_size, input_size)


def test_ncde_initial_insert_at_zero():
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


@pytest.mark.parametrize("inference", [True, False])
def test_mlpneuralcde(inference: bool):
    key = jr.key(7)
    in_size, out_size, latent_size, state_size = 4, 2, 6, 4
    model, state = eqx.nn.make_with_state(MLPNeuralCDE)(
        in_size=in_size,
        out_size=out_size,
        latent_size=latent_size,
        width_size=7,
        depth=1,
        key=key,
        state_size=state_size,
        inference=inference,
    )
    state, ys = model(state, jnp.zeros(()), jnp.zeros((in_size,)))
    state, ys = model(state, jnp.ones(()), jnp.ones((in_size,)))

    assert isinstance(ys, jnp.ndarray)
    assert ys.shape == (out_size,)
    assert isinstance(model.t1(state), jnp.ndarray)
    assert model.t1(state).shape == ()
    assert model.t1(state) == 1.0
    assert isinstance(model.ts(state), jnp.ndarray)
    assert model.ts(state).shape == (state_size,)
    assert isinstance(model.x1(state), jnp.ndarray)
    assert model.x1(state).shape == (in_size,)
    assert isinstance(model.xs(state), jnp.ndarray)
    assert model.xs(state).shape == (state_size, in_size)
    assert isinstance(model.z1(state), jnp.ndarray)
    assert model.z1(state).shape == (latent_size,)
    assert isinstance(model.zs(state), jnp.ndarray)
    assert model.zs(state).shape == (state_size, latent_size)


@pytest.mark.parametrize("time_in_input", [True, False])
def test_ncde_z0_and_coeffs_shapes(time_in_input: bool):
    key = jr.key(0)
    in_size, latent_size = 3, 5
    model = MLPNeuralCDE(
        in_size=in_size,
        out_size=2,
        latent_size=latent_size,
        width_size=4,
        depth=1,
        key=key,
        time_in_input=time_in_input,
    )

    t0 = jnp.array(0.0)
    x0 = jnp.ones((in_size,))
    z0 = model.z0(t0, x0)
    assert z0.shape == (latent_size,)

    ts = jnp.linspace(0.0, 1.0, 6)
    xs = jnp.zeros((6, in_size))
    coeffs = model.coeffs(ts, xs)
    expected_channels = in_size + int(time_in_input)
    assert len(coeffs) == 4
    for c in coeffs:
        assert c.shape == (len(ts) - 1, expected_channels)


@pytest.mark.parametrize("inference", [True, False])
def test_ncde_solve_output_shape(inference: bool):
    key = jr.key(1)
    in_size, latent_size = 2, 4
    model = MLPNeuralCDE(
        in_size=in_size,
        out_size=1,
        latent_size=latent_size,
        width_size=4,
        depth=1,
        key=key,
        solver=diffrax.Euler,
        inference=inference,
    )

    ts = jnp.linspace(0.0, 1.0, 5)
    xs = jnp.zeros((5, in_size))
    coeffs = model.coeffs(ts, xs)
    z0 = model.z0(ts[0], xs[0])
    zs = model.solve(ts, z0, coeffs)
    assert zs.shape == (len(ts), latent_size)


@pytest.mark.parametrize("inference", [True, False])
def test_ncde_state_rollover(inference: bool):
    key = jr.key(2)
    in_size, out_size, latent_size = 2, 1, 3
    model, state = eqx.nn.make_with_state(MLPNeuralCDE)(
        in_size=in_size,
        out_size=out_size,
        latent_size=latent_size,
        width_size=4,
        depth=1,
        key=key,
        state_size=3,
        solver=diffrax.Euler,
        inference=inference,
    )

    for t in jnp.arange(4.0):
        x = jnp.full((in_size,), t)
        state, y = model(state, t, x)
        assert y.shape == (out_size,)

    ts = model.ts(state)
    assert not jnp.isnan(ts).any()
    assert jnp.allclose(ts, jnp.array([1.0, 2.0, 3.0]))
    assert model.t1(state) == 3.0


@pytest.mark.parametrize("dims", range(4))
def test_flatten_model(dims: int):
    model = Flatten()

    input_shape = (2,) * dims
    input_data = jnp.ones(input_shape)

    output = model(input_data)

    assert isinstance(output, jnp.ndarray)
    assert output.shape == (2**dims,)
    assert jnp.array_equal(output, jnp.ones((2**dims,)))
