from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Bool, Float, Key

from oryx.spaces import AbstractSpace


class AbstractEnvLike[ActType, ObsType](eqx.Module):
    """Base class for RL environments or wrappers that behave like environments"""

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]

    @abstractmethod
    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        """Reset the environment to an initial state"""

    @abstractmethod
    def step(
        self, state: eqx.nn.State, action: ActType, *, key: Key
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        """
        Perform a step of the environment

        :param state: The environment state
        :param action: The action
        :param key: Random key
        :returns: The new environment state, the observation, the reward, termination
            flag, truncation flag, and info dict
        """

    @abstractmethod
    def render(self, state: eqx.nn.State):
        """Render a frame from a state"""

    @abstractmethod
    def close(self):
        """Close the environment"""

    @property
    @abstractmethod
    def action_space(self) -> AbstractSpace[ActType]:
        """Return the action space of the environment"""

    @property
    @abstractmethod
    def observation_space(self) -> AbstractSpace[ObsType]:
        """Return the observation space of the environment"""

    @property
    @abstractmethod
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        """Return the unwrapped environment"""


class AbstractEnv[ActType, ObsType](AbstractEnvLike[ActType, ObsType]):
    """Base class for RL environments"""

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]

    @property
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        """Return the unwrapped environment"""
        return self
