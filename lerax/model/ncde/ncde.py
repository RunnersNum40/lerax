from __future__ import annotations

from typing import Callable

import diffrax
import equinox as eqx
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, PyTree, Shaped

from ..base_model import AbstractModelState, AbstractStatefulModel
from .term import AbstractNCDETerm, MLPNCDETerm
from .utils import safe_control

type Coeffs = tuple[
    PyTree[Shaped[Array, "times-1 ?*shape"]],
    PyTree[Shaped[Array, "times-1 ?*shape"]],
    PyTree[Shaped[Array, "times-1 ?*shape"]],
    PyTree[Shaped[Array, "times-1 ?*shape"]],
]


class NCDEState(AbstractModelState):
    ts: Float[Array, " n"]
    xs: Float[Array, " n input_size"]


class AbstractNeuralCDE[
    LatentType: Callable[[Float[Array, " out_size"]], Float[Array, " latent_size"]]
](
    AbstractStatefulModel[
        NCDEState,
        [Float[Array, ""], Float[Array, " in_size"]],
        Float[Array, " out_size"],
    ],
):
    """
    Abstract base class for Neural Controlled Differential Equations (NCDEs).

    x is used to denote the input, t is the time, z is the latent state, and y is the
    output. _s is used to denote all states of a variable (e.g. zs is the latent states
    at all times) and _1 is used to denote the last state of a variable (e.g. z1 is the
    latent state at the last time).

    The state_size defines the maximum number of inputs and corresponding states that
    are tracked. In theory the latest latent vector should contain all the important
    information about the history of inputs and states, but longer history
    allows more accurate gradients to be computed for back propagation through time.
    Inference mode can be used to disable the history and only use the latest state for
    faster computation without back propagation through time.
    """

    term: eqx.AbstractVar[AbstractNCDETerm]
    solver: eqx.AbstractVar[diffrax.AbstractSolver]

    in_size: eqx.AbstractVar[int]
    latent_size: eqx.AbstractVar[int]

    initial: eqx.AbstractVar[LatentType]
    time_in_input: eqx.AbstractVar[bool]

    history_length: eqx.AbstractVar[int]

    def solve(
        self,
        ts: Float[Array, " n"],
        xs: Float[Array, " n in_size"],
    ) -> Float[Array, " n latent_size"]:
        if isinstance(self.solver, diffrax.AbstractAdaptiveSolver):
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
            dt0 = None
        else:
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = jnp.nanmean(jnp.diff(ts))

        z0 = self.z0(ts[0], xs[0])
        control = safe_control(ts, xs)
        term = diffrax.ControlTerm(self.term, control).to_ode()

        solution = diffrax.diffeqsolve(
            terms=term,
            solver=self.solver,
            t0=ts[0],
            t1=jnp.nanmax(ts),
            dt0=dt0,
            y0=z0,
            stepsize_controller=stepsize_controller,
        )

        assert solution.ys is not None
        return solution.ys[0]

    def z0(
        self, t0: Float[Array, ""], x0: Float[Array, " in_size"]
    ) -> Float[Array, " latent_size"]:
        if self.time_in_input:
            return self.initial(
                jnp.concatenate([x0, jnp.expand_dims(t0, axis=-1)], axis=-1)
            )
        else:
            return self.initial(x0)

    def next_state(
        self, state: NCDEState, ti: Float[Array, ""], xi: Float[Array, " in_size"]
    ) -> tuple[Float[Array, " num_steps"], Float[Array, " num_steps input_size"]]:
        """Add new time and input pair to the state."""
        ts, xs = state.ts, state.xs
        latest_index = jnp.nanargmax(ts)
        ts = eqx.error_if(
            ts,
            ti <= ts[latest_index],
            "new input and time must be later than all previous",
        )

        def shift() -> (
            tuple[Float[Array, " num_steps"], Float[Array, " num_steps input_size"]]
        ):
            """Shift the saved times and inputs to make room for the new pair."""
            return (
                jnp.roll(ts, -1).at[-1].set(ti),
                jnp.roll(xs, -1, axis=0).at[-1].set(xi),
            )

        def insert() -> (
            tuple[Float[Array, " num_steps"], Float[Array, " num_steps input_size"]]
        ):
            """Insert the new time and input pair at the end of the saved times and inputs."""
            return ts.at[latest_index + 1].set(ti), xs.at[latest_index + 1].set(xi)

        ts, xs = lax.cond(latest_index == self.history_length - 1, shift, insert)

        return ts, xs

    def __call__(
        self,
        state: NCDEState,
        ti: Float[Array, ""],
        xi: Float[Array, " in_size"],
    ) -> tuple[NCDEState, Float[Array, " out_size"]]:
        """Compute the next state and output given the current state and input."""
        ts, xs = self.next_state(state, ti, xi)
        zi = self.solve(ts, xs)
        return NCDEState(ts, xs), zi

    def reset(self) -> NCDEState:
        """Reset the state to an empty state."""
        times = jnp.full((self.history_length,), jnp.nan, dtype=float)
        inputs = jnp.full((self.history_length, self.in_size), jnp.nan, dtype=float)
        return NCDEState(times, inputs)


class MLPNeuralCDE(AbstractNeuralCDE):
    term: MLPNCDETerm
    solver: diffrax.AbstractSolver

    initial: eqx.nn.MLP
    time_in_input: bool

    in_size: int
    latent_size: int

    history_length: int

    def __init__(
        self,
        in_size: int,
        latent_size: int,
        *,
        term: MLPNCDETerm | None = None,
        initial: eqx.nn.MLP | None = None,
        field_width: int = 64,
        field_depth: int = 2,
        field_activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.softplus,
        field_final_activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.tanh,
        initial_width: int = 64,
        initial_depth: int = 1,
        initial_state_activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.relu,
        initial_state_final_activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = lambda x: x,
        solver: diffrax.AbstractSolver | None = None,
        time_in_input: bool = False,
        history_length: int = 16,
        key: Key,
    ):
        term_key, initial_key = jr.split(key, 2)

        self.solver = solver or diffrax.Tsit5()
        self.time_in_input = time_in_input
        self.history_length = history_length

        self.in_size = in_size
        self.latent_size = latent_size

        self.term = (
            term
            if term is not None
            else MLPNCDETerm(
                input_size=in_size,
                state_size=latent_size,
                width_size=field_width,
                depth=field_depth,
                key=term_key,
                add_time=time_in_input,
                activation=field_activation,
                final_activation=field_final_activation,
            )
        )

        self.initial = (
            initial
            if initial is not None
            else eqx.nn.MLP(
                in_size=in_size + int(time_in_input),
                out_size=latent_size,
                width_size=initial_width,
                depth=initial_depth,
                key=initial_key,
                activation=initial_state_activation,
                final_activation=initial_state_final_activation,
            )
        )
