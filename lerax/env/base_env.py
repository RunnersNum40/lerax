from __future__ import annotations

import time
from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Bool, Float, Key

from lerax.render import AbstractRenderer
from lerax.space import AbstractSpace
from lerax.utils import unstack_pytree


class AbstractEnvLikeState(eqx.Module):

    @property
    @abstractmethod
    def unwrapped(self) -> AbstractEnvState:
        """Return the unwrapped environment state"""


class AbstractEnvLike[StateType: AbstractEnvLikeState, ActType, ObsType](eqx.Module):
    """Base class for RL environments or wrappers that behave like environments"""

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def reset(self, *, key: Key) -> tuple[StateType, ObsType, dict]:
        """Reset the environment to an initial state"""

    @abstractmethod
    def step(
        self, state: StateType, action: ActType, *, key: Key
    ) -> tuple[
        StateType, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        """Perform a step of the environment"""

    @abstractmethod
    def render(self, state: StateType):
        """Render a frame from a state"""

    def render_stacked(self, states: StateType, dt: float = 0.0):
        """Render multiple frames from stacked states"""
        inner_state = states.unwrapped
        unstacked_states = unstack_pytree(inner_state)
        for state in unstacked_states:
            self.unwrapped.render(state)
            time.sleep(dt)

    @abstractmethod
    def close(self):
        """Close the environment"""

    @property
    @abstractmethod
    def unwrapped(self) -> AbstractEnv:
        """Return the unwrapped environment"""


class AbstractEnvState(AbstractEnvLikeState):
    @property
    def unwrapped(self) -> AbstractEnvState:
        """Return the unwrapped environment state"""
        return self


class AbstractEnv[StateType: AbstractEnvState, ActType, ObsType](
    AbstractEnvLike[StateType, ActType, ObsType]
):
    """Base class for RL environments"""

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    renderer: eqx.AbstractVar[AbstractRenderer | None]

    @property
    def unwrapped(self) -> AbstractEnv[StateType, ActType, ObsType]:
        """Return the unwrapped environment"""
        return self
