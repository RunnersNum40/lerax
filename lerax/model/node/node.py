from __future__ import annotations

from typing import Callable

import diffrax
import equinox as eqx
import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key

from ..base_model import AbstractModel
from .term import AbstractNODETerm, MLPNODETerm


class AbstractNeuralODE(
    AbstractModel[
        [Float[Array, " n"], Float[Array, " data_size"]], Float[Array, " n data_size"]
    ],
):
    """Neural ODE model"""

    term: eqx.AbstractVar[AbstractNODETerm]
    solver: eqx.AbstractVar[type[diffrax.AbstractSolver]]

    def solve(
        self, ts: Float[Array, " n"], z0: Float[Array, " data_size"]
    ) -> Float[Array, " n data_size"]:
        solver = self.solver()
        if isinstance(solver, diffrax.AbstractAdaptiveSolver):
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
            dt0 = None
        else:
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = jnp.diff(ts).mean()

        term = diffrax.ODETerm(self.term)
        saveat = diffrax.SaveAt(ts=ts)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=jnp.min(ts),
            t1=jnp.max(ts),
            dt0=dt0,
            y0=z0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )

        assert solution.ys is not None

        return solution.ys


class AbstractLatentNeuralODE[
    T: Callable[[Array], Float[Array, " latent_size"]],
    P: Callable[[Array], Float[Array, " latent_size"]],
](AbstractNeuralODE):

    in_size: eqx.AbstractVar[int]
    latent_size: eqx.AbstractVar[int]
    out_size: eqx.AbstractVar[int]

    initial: eqx.AbstractVar[T]
    output: eqx.AbstractVar[P]
    time_in_input: eqx.AbstractVar[bool]

    def __call__(
        self, ts: Float[Array, " n"], x0: Float[Array, " in_size"]
    ) -> Float[Array, " n output_size"]:
        if self.time_in_input:
            z0 = self.initial(
                jnp.concatenate([x0, jnp.expand_dims(ts[0], axis=-1)], axis=-1)
            )
        else:
            z0 = self.initial(x0)

        return jax.vmap(self.output)(self.solve(ts, z0))


class MLPNeuralODE(AbstractLatentNeuralODE[eqx.nn.MLP, eqx.nn.MLP]):
    """MLP based neural ODE."""

    term: MLPNODETerm
    solver: type[diffrax.AbstractSolver]

    in_size: int
    latent_size: int
    out_size: int

    initial: eqx.nn.MLP
    output: eqx.nn.MLP
    time_in_input: bool

    def __init__(
        self,
        in_size: int,
        out_size: int,
        latent_size: int,
        width_size: int,
        depth: int,
        activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.softplus,
        final_activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.tanh,
        *,
        key: Key,
        solver: type[diffrax.AbstractSolver] = diffrax.Tsit5,
        time_in_input: bool = True,
    ):
        term_key, initial_key, output_key = jr.split(key, 3)

        self.solver = solver
        self.time_in_input = time_in_input

        self.in_size = in_size
        self.latent_size = latent_size
        self.out_size = out_size

        self.term = MLPNODETerm(
            data_size=latent_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=term_key,
            add_time=time_in_input,
        )
        self.initial = eqx.nn.MLP(
            in_size=in_size + int(time_in_input),
            out_size=latent_size,
            width_size=width_size,
            depth=depth,
            key=initial_key,
        )
        self.output = eqx.nn.MLP(
            in_size=latent_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=output_key,
        )
