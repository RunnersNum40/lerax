from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Bool, Float, Key

from oryx.space import AbstractSpace


class AbstractEnvLike[ActType, ObsType](eqx.Module):
    """Base class for RL environments or wrappers that behave like environments"""

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

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
        """

    @abstractmethod
    def render(self, state: eqx.nn.State):
        """Render a frame from a state"""

    @abstractmethod
    def close(self):
        """Close the environment"""

    @property
    @abstractmethod
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        """Return the unwrapped environment"""


class AbstractEnv[ActType, ObsType](AbstractEnvLike[ActType, ObsType]):
    """Base class for RL environments"""

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @property
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        """Return the unwrapped environment"""
        return self
