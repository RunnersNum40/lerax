from functools import partial

import diffrax
import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jaxtyping import Array, Float, ScalarLike

from lerax.model.ncde import AbstractNCDETerm, AbstractNeuralCDE, NCDEState


class SimpleNCDETerm(AbstractNCDETerm):
    fill: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.0))

    state_size: int = 2
    input_size: int = 2

    def __call__(
        self, t: ScalarLike, z: Float[Array, " state_size"], args
    ) -> Float[Array, " state_size input_size"]:
        """Vector field value."""

        return jnp.full((self.state_size, self.input_size), self.fill)


class Scale(eqx.Module):
    scale: Float[Array, "..."] = eqx.field(converter=jnp.asarray)

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x * self.scale


class SimpleNeuralCDE(AbstractNeuralCDE):
    term: SimpleNCDETerm = SimpleNCDETerm()

    in_size: int = 2
    latent_size: int = 2
    history_length: int = 16

    time_in_input: bool = False

    initial: Scale = Scale(scale=5.0)

    solver: diffrax.AbstractSolver = diffrax.Tsit5()


class TestNeuralCDE:
    @pytest.fixture
    def model(self) -> SimpleNeuralCDE:
        return SimpleNeuralCDE()

    @staticmethod
    def x(t: Float[Array, ""]) -> Float[Array, "1"]:
        return jnp.array([jnp.cos(t), jnp.sin(t)])

    def test_initial_state(self, model: SimpleNeuralCDE):
        state = model.reset()

        assert isinstance(state, NCDEState)
        assert jnp.isnan(state.ts).all()
        assert jnp.isnan(state.xs).all()

        t0 = jnp.array(0.0)
        x0 = self.x(t0)
        model(state, t0, x0)

    def test_grad_first_step(self, model: SimpleNeuralCDE):
        @eqx.filter_value_and_grad
        def model_grad(model, state, t0, x0):
            return jnp.sum(model(state, t0, x0)[1])

        @eqx.filter_value_and_grad
        def expected_grad(model, t0, x0):
            return jnp.sum(model.initial(x0))

        state = model.reset()
        t0 = jnp.array(0.0)
        x0 = self.x(t0)

        model_output, model_grads = model_grad(model, state, t0, x0)
        expected_output, expected_grads = expected_grad(model, t0, x0)

        assert jnp.array_equal(model_output, expected_output)
        assert eqx.tree_equal(model_grads, expected_grads)

    def test_grad_full_integration(self, model: SimpleNeuralCDE):
        ts = jnp.linspace(0.0, 2 * jnp.pi, num=model.history_length)
        xs = jax.vmap(self.x)(ts)

        @eqx.filter_value_and_grad(has_aux=True)
        def model_grad(model, state, ti, xi):
            state, yi = model(state, ti, xi)
            return jnp.sum(yi), state

        @eqx.filter_value_and_grad(has_aux=True)
        def expected_grad(model, state, ti, xi):
            coeffs = diffrax.backward_hermite_coefficients(ts, xs)
            control = diffrax.CubicInterpolation(ts, coeffs)
            term = diffrax.ControlTerm(model.term, control).to_ode()

            solution = diffrax.diffeqsolve(
                terms=term,
                solver=model.solver,
                t0=ts[0],
                t1=ti,
                dt0=None,
                y0=model.initial(xs[0]),
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                saveat=diffrax.SaveAt(t1=True),
            )

            assert solution.ys is not None
            zi = solution.ys[0]

            li = jnp.nanargmax(state.ts)
            # Only correct for up to `model.history_length` steps
            state = NCDEState(
                ts=state.ts.at[li].set(ti),
                xs=state.xs.at[li].set(xi),
            )

            return jnp.sum(zi), state

        state = model.reset()

        for ti, xi in zip(ts, xs):
            (model_output, state), model_grads = model_grad(model, state, ti, xi)
            (expected_output, expected_state), expected_grads = expected_grad(
                model, state, ti, xi
            )

            allclose = partial(jnp.allclose, rtol=1e-5, atol=1e-8, equal_nan=True)

            assert jax.tree.reduce(
                jnp.logical_and,
                jax.tree.map(lambda leaf: jnp.all(jnp.isfinite(leaf)), model_grads),
            )

            assert allclose(model_output, expected_output)
            # eqx.tree_equal does not support equal_nan=True
            assert allclose(state.ts, expected_state.ts)
            assert allclose(state.xs, expected_state.xs)

            assert eqx.tree_equal(model_grads, expected_grads, rtol=1e-5, atol=1e-8)
