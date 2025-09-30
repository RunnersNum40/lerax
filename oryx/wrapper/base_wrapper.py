from __future__ import annotations

import equinox as eqx

from oryx.env import AbstractEnv, AbstractEnvLike
from oryx.space import AbstractSpace


class AbstractWrapper[WrapperActType, WrapperObsType, ActType, ObsType](
    AbstractEnvLike[WrapperActType, WrapperObsType]
):
    """Base class for environment wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    @property
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        """Return the unwrapped environment"""
        return self.env.unwrapped

    @property
    def name(self) -> str:
        """Return the name of the environment"""
        return self.env.name


class AbstractNoRenderWrapper[WrapperActType, WrapperObsType, ActType, ObsType](
    AbstractWrapper[WrapperActType, WrapperObsType, ActType, ObsType]
):
    """A wrapper that does not affect rendering"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def render(self, state: eqx.nn.State):
        return self.env.render(state)


class AbstractNoCloseWrapper[WrapperActType, WrapperObsType, ActType, ObsType](
    AbstractWrapper[WrapperActType, WrapperObsType, ActType, ObsType]
):
    """A wrapper that does not affect closing"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def close(self):
        return self.env.close()


class AbstractNoRenderOrCloseWrapper[WrapperActType, WrapperObsType, ActType, ObsType](
    AbstractNoRenderWrapper[WrapperActType, WrapperObsType, ActType, ObsType],
    AbstractNoCloseWrapper[WrapperActType, WrapperObsType, ActType, ObsType],
):
    """A wrapper that does not affect rendering or closing the environment"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]


class AbstractNoActionSpaceWrapper[WrapperObsType, ActType, ObsType](
    AbstractWrapper[ActType, WrapperObsType, ActType, ObsType]
):
    """A wrapper that does not affect the action space"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space


class AbstractNoObservationSpaceWrapper[WrapperActType, ActType, ObsType](
    AbstractWrapper[WrapperActType, ObsType, ActType, ObsType]
):
    """A wrapper that does not affect the observation space"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space


class AbstractNoActionOrObservationSpaceWrapper[ActType, ObsType](
    AbstractNoActionSpaceWrapper[ObsType, ActType, ObsType],
    AbstractNoObservationSpaceWrapper[ActType, ActType, ObsType],
):
    """A wrapper that does not affect the action or observation space"""
