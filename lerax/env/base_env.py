from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Bool, Float, Key

from lerax.space import AbstractSpace


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

    @property
    def unwrapped(self) -> AbstractEnv[StateType, ActType, ObsType]:
        """Return the unwrapped environment"""
        return self
