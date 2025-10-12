from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from lerax.env import AbstractEnvLike

from .base_wrapper import (
    AbstractNoActionOrObservationSpaceWrapper,
    AbstractNoRenderOrCloseWrapper,
)


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


class AbstractPureTransformRewardWrapper[ActType, ObsType](
    AbstractTransformRewardWrapper[ActType, ObsType]
):
    """
    Apply a *pure* (stateless) function to every reward emitted by the wrapped
    environment.
    """

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[Float[Array, ""]], Float[Array, ""]]]

    def reward(
        self,
        state: eqx.nn.State,
        reward: Float[Array, ""],
        *,
        key: Key,
    ) -> Float[Array, ""]:
        return self.func(reward)


class ClipReward[ActType, ObsType](
    AbstractPureTransformRewardWrapper[ActType, ObsType]
):
    """
    Element-wise clip of rewards:  `reward ↦ clamp(min, max)`.
    """

    env: AbstractEnvLike[ActType, ObsType]
    func: Callable[[Float[Array, ""]], Float[Array, ""]]
    min: Float[Array, ""]
    max: Float[Array, ""]

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        min: Float[ArrayLike, ""] = jnp.asarray(-1.0),
        max: Float[ArrayLike, ""] = jnp.asarray(1.0),
    ):
        self.env = env
        self.min = jnp.asarray(min)
        self.max = jnp.asarray(max)
        self.func = lambda r: jnp.clip(r, self.min, self.max)
