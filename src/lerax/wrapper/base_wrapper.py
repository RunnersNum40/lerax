from __future__ import annotations

from typing import Any

import equinox as eqx

from lerax.env import (
    AbstractEnv,
    AbstractEnvLike,
    AbstractEnvLikeState,
    AbstractEnvState,
)
from lerax.render import AbstractRenderer
from lerax.space import AbstractSpace


class AbstractWrapperState[StateType: AbstractEnvLikeState](AbstractEnvLikeState):
    env_state: eqx.AbstractVar[StateType]

    @property
    def unwrapped(self) -> AbstractEnvState:
        """The state of the wrapped environment"""
        return self.env_state.unwrapped


class AbstractWrapper[
    WrapperStateType: AbstractWrapperState,
    WrapperActType,
    WrapperObsType,
    WrapperMaskType,
    StateType: AbstractEnvLikeState,
    ActType,
    ObsType,
    MaskType,
](AbstractEnvLike[WrapperStateType, WrapperActType, WrapperObsType, WrapperMaskType]):
    """
    Base class for environment wrappers.

    Attributes:
        name: The name of the environment
        env: The wrapped environment
        unwrapped: The environment without any wrappers
        action_space: The action space of the environment after wrapping
        observation_space: The observation space of the environment after wrapping
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType, MaskType]]

    action_space: eqx.AbstractVar[AbstractSpace[ActType, WrapperMaskType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType, Any]]

    @property
    def unwrapped(self) -> AbstractEnv:
        """Return the wrapped environment"""
        return self.env.unwrapped

    @property
    def name(self) -> str:
        """Return the name of the environment"""
        return self.env.name

    def default_renderer(self) -> AbstractRenderer:
        """Return the default renderer for the wrapped environment"""
        return self.unwrapped.default_renderer()

    def render(self, state: WrapperStateType, renderer: AbstractRenderer):
        """Render a frame from a state"""
        self.unwrapped.render(state.unwrapped, renderer)
