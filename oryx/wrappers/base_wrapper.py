from abc import abstractmethod

import equinox as eqx
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from oryx.env import AbstractEnv, AbstractEnvLike
from oryx.spaces import AbstractSpace


class AbstractWrapper[WrapperActType, WrapperObsType, ActType, ObsType](
    AbstractEnvLike[WrapperActType, WrapperObsType]
):
    """Base class for environment wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    @property
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        """Return the unwrapped environment"""
        return self.env.unwrapped


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


class AbstractTransformObservationWrapper[WrapperObsType, ActType, ObsType](
    AbstractNoRenderOrCloseWrapper[ActType, WrapperObsType, ActType, ObsType],
    AbstractNoActionSpaceWrapper[WrapperObsType, ActType, ObsType],
):
    """Base class for environment observation wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, WrapperObsType, dict]:
        env_key, wrapper_key = jr.split(key, 2)

        substate = state.substate(self.env)
        substate, obs, info = self.env.reset(substate, key=env_key)
        state, obs = self.observation(state, obs, key=wrapper_key)

        state = state.update(substate)

        return state, obs, info

    def step(self, state: eqx.nn.State, action: ActType, *, key: Key) -> tuple[
        eqx.nn.State,
        WrapperObsType,
        Float[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        dict,
    ]:
        env_key, wrapper_key = jr.split(key, 2)

        env_state = state.substate(self.env)
        env_state, obs, reward, termination, truncation, info = self.env.step(
            env_state, action, key=env_key
        )
        state, obs = self.observation(state, obs, key=wrapper_key)

        state = state.update(env_state)

        return state, obs, reward, termination, truncation, info

    @abstractmethod
    def observation(
        self, state: eqx.nn.State, obs: ObsType, *, key: Key
    ) -> tuple[eqx.nn.State, WrapperObsType]:
        """Transform the wrapped environment observation"""


class AbstractTransformActionWrapper[WrapperActType, ActType, ObsType](
    AbstractNoRenderOrCloseWrapper[WrapperActType, ObsType, ActType, ObsType],
    AbstractNoObservationSpaceWrapper[WrapperActType, ActType, ObsType],
):
    """Base class for environment action wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        substate = state.substate(self.env)
        substate, obs, info = self.env.reset(substate, key=key)
        state = state.update(substate)

        return state, obs, info

    def step(
        self, state: eqx.nn.State, action: WrapperActType, *, key: Key
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        env_key, wrapper_key = jr.split(key, 2)

        state, transformed_action = self.action(state, action, key=wrapper_key)

        env_state = state.substate(self.env)
        env_state, obs, reward, termination, truncation, info = self.env.step(
            env_state, transformed_action, key=env_key
        )
        state = state.update(env_state)

        return state, obs, reward, termination, truncation, info

    @abstractmethod
    def action(
        self, state: eqx.nn.State, action: WrapperActType, *, key: Key
    ) -> tuple[eqx.nn.State, ActType]:
        """Transform the action to the wrapped environment"""


class AbstractTransformRewardWrapper[ActType, ObsType](
    AbstractNoRenderOrCloseWrapper[ActType, ObsType, ActType, ObsType],
    AbstractNoActionOrObservationSpaceWrapper[ActType, ObsType],
):
    """Base class for environment reward wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        substate = state.substate(self.env)
        substate, obs, info = self.env.reset(substate, key=key)
        state = state.update(substate)

        return state, obs, info

    def step(
        self, state: eqx.nn.State, action: ActType, *, key: Key
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        env_key, wrapper_key = jr.split(key, 2)

        env_state = state.substate(self.env)
        env_state, obs, reward, termination, truncation, info = self.env.step(
            env_state, action, key=env_key
        )
        state = state.update(env_state)

        reward = self.reward(state, reward, key=wrapper_key)

        return state, obs, reward, termination, truncation, info

    @abstractmethod
    def reward(
        self, state: eqx.nn.State, reward: Float[Array, ""], *, key: Key
    ) -> Float[Array, ""]:
        """Transform the reward from the wrapped environment"""
