from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from lerax.env import AbstractEnvLike
from lerax.space import AbstractSpace, Box

from .base_wrapper import (
    AbstractNoObservationSpaceWrapper,
    AbstractNoRenderOrCloseWrapper,
)
from .utils import rescale_box


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


class AbstractPureTransformActionWrapper[WrapperActType, ActType, ObsType](
    AbstractTransformActionWrapper[WrapperActType, ActType, ObsType]
):
    """
    Base class for wrappers that apply a pure function to the action before passing it to
    the environment
    """

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[WrapperActType], ActType]]
    action_space: eqx.AbstractVar[AbstractSpace[WrapperActType]]

    def action(
        self, state: eqx.nn.State, action: WrapperActType, *, key: Key
    ) -> tuple[eqx.nn.State, ActType]:
        transformed_action = self.func(action)
        return state, transformed_action


class TransformAction[WrapperActType, ActType, ObsType](
    AbstractPureTransformActionWrapper[WrapperActType, ActType, ObsType]
):
    """Apply a function to the action before passing it to the environment"""

    env: AbstractEnvLike[ActType, ObsType]
    func: Callable[[WrapperActType], ActType]
    action_space: AbstractSpace[WrapperActType]

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        func: Callable[[WrapperActType], ActType],
        action_space: AbstractSpace[WrapperActType],
    ):
        self.env = env
        self.func = func
        self.action_space = action_space


class ClipAction[ObsType](
    AbstractPureTransformActionWrapper[
        Float[Array, " ..."], Float[Array, " ..."], ObsType
    ],
):
    """
    Clips every action to the environment's action space.
    """

    env: AbstractEnvLike[Float[Array, " ..."], ObsType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    action_space: Box

    def __init__(self, env: AbstractEnvLike[Float[Array, " ..."], ObsType]):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                "ClipAction only supports `Box` action spaces "
                f"not {type(env.action_space)}"
            )

        def clip(action: Float[Array, " ..."]) -> Float[Array, " ..."]:
            assert isinstance(env.action_space, Box)
            return jnp.clip(action, env.action_space.low, env.action_space.high)

        action_space = Box(-jnp.inf, jnp.inf, shape=env.action_space.shape)

        self.env = env
        self.func = clip
        self.action_space = action_space


class RescaleAction[ObsType](
    AbstractPureTransformActionWrapper[
        Float[Array, " ..."], Float[Array, " ..."], ObsType
    ],
):
    """Affinely rescale a box action to a different range"""

    env: AbstractEnvLike[Float[Array, " ..."], ObsType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    action_space: Box

    def __init__(
        self,
        env: AbstractEnvLike[Float[Array, " ..."], ObsType],
        min: Float[Array, " ..."] = jnp.array(-1.0),
        max: Float[Array, " ..."] = jnp.array(1.0),
    ):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                "RescaleActiononly supports `Box` action spaces"
                f" not {type(env.action_space)}"
            )

        action_space, _, rescale = rescale_box(env.action_space, min, max)

        self.env = env
        self.func = rescale
        self.action_space = action_space
