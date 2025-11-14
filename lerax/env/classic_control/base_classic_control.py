from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace

import diffrax
import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from lerax.space import AbstractSpace

from ..base_env import AbstractEnv, AbstractEnvState


class AbstractClassicControlEnvState(AbstractEnvState):
    y: eqx.AbstractVar[Float[Array, "..."]]
    t: eqx.AbstractVar[Float[Array, ""]]


class AbstractClassicControlEnv[
    StateType: AbstractClassicControlEnvState, ActType, ObsType
](AbstractEnv[StateType, ActType, ObsType]):
    name: eqx.AbstractClassVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    dt: eqx.AbstractVar[float]
    solver: eqx.AbstractVar[diffrax.AbstractSolver]
    dt0: eqx.AbstractVar[float | None]
    stepsize_controller: eqx.AbstractVar[diffrax.AbstractStepSizeController]

    @abstractmethod
    def dynamics(
        self, t: Float[Array, ""], y: Float[Array, " n"], action: ActType
    ) -> Float[Array, " n"]:
        """Compute the time derivative of the state."""

    @abstractmethod
    def clip(self, y: Float[Array, " n"]) -> Float[Array, " n"]:
        """Clip the state to be within valid bounds."""

    def transition(self, state: StateType, action: ActType, *, key: Key) -> StateType:
        @diffrax.ODETerm
        def term(t, y, args):
            return self.dynamics(t, y, action)

        saveat = diffrax.SaveAt(t1=True)
        sol = diffrax.diffeqsolve(
            term,
            solver=self.solver,
            t0=state.t,
            t1=state.t + self.dt,
            dt0=self.dt0,
            y0=state.y,
            args=action,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
        )
        assert sol.ys is not None

        return replace(state, y=self.clip(sol.ys[0]), t=state.t + self.dt)

    def truncate(self, state: StateType) -> Bool[Array, ""]:
        return jnp.array(False)

    def state_info(self, state: StateType) -> dict:
        return {}

    def transition_info(
        self,
        state: StateType,
        action: ActType,
        next_state: StateType,
    ) -> dict:
        return {}
