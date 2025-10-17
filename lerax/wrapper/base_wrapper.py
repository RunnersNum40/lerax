from __future__ import annotations

import equinox as eqx

from lerax.env import (
    AbstractEnv,
    AbstractEnvLike,
    AbstractEnvLikeState,
    AbstractEnvState,
)


class AbstractWrapperState[StateType: AbstractEnvLikeState](AbstractEnvLikeState):
    env_state: eqx.AbstractVar[StateType]

    @property
    def unwrapped(self) -> AbstractEnvState:
        return self.env_state.unwrapped


class AbstractWrapper[
    WrapperStateType: AbstractEnvLikeState,
    WrapperActType,
    WrapperObsType,
    StateType: AbstractEnvLikeState,
    ActType,
    ObsType,
](AbstractEnvLike[WrapperStateType, WrapperActType, WrapperObsType]):
    """Base class for environment wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType]]

    @property
    def unwrapped(self) -> AbstractEnv:
        """Return the unwrapped environment"""
        return self.env.unwrapped

    @property
    def name(self) -> str:
        """Return the name of the environment"""
        return self.env.name
