from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from lerax.env import AbstractEnvLike
from lerax.space import AbstractSpace, Box

from .base_wrapper import (
    AbstractNoActionSpaceWrapper,
    AbstractNoRenderOrCloseWrapper,
)
from .utils import rescale_box


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


class AbstractPureObservationWrapper[WrapperObsType, ActType, ObsType](
    AbstractTransformObservationWrapper[WrapperObsType, ActType, ObsType]
):
    """
    Apply a pure function to every observation that leaves the environment.
    """

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[ObsType], WrapperObsType]]
    observation_space: eqx.AbstractVar[AbstractSpace[WrapperObsType]]

    def observation(
        self, state: eqx.nn.State, obs: ObsType, *, key: Key
    ) -> tuple[eqx.nn.State, WrapperObsType]:
        return state, self.func(obs)


class ClipObservation[ActType](
    AbstractPureObservationWrapper[Float[Array, " ..."], ActType, Float[Array, " ..."]],
):
    """
    Clips every observation to the environment's observation space.
    """

    env: AbstractEnvLike[ActType, Float[Array, " ..."]]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    observation_space: Box

    def __init__(self, env: AbstractEnvLike[ActType, Float[Array, " ..."]]):
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                "ClipObservation only supports `Box` observation spaces "
                f" not {type(env.observation_space)}"
            )

        self.env = env
        self.func = partial(
            jnp.clip,
            min=env.observation_space.low,
            max=env.observation_space.high,
        )
        self.observation_space = env.observation_space


class RescaleObservation[ActType](
    AbstractPureObservationWrapper[Float[Array, " ..."], ActType, Float[Array, " ..."]],
):
    """Affinely rescale a box observation to a different range"""

    env: AbstractEnvLike[ActType, Float[Array, " ..."]]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    observation_space: Box

    def __init__(
        self,
        env: AbstractEnvLike[ActType, Float[Array, " ..."]],
        min: Float[Array, " ..."] = jnp.array(-1.0),
        max: Float[Array, " ..."] = jnp.array(1.0),
    ):
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                "RescaleObservation only supports `Box` observation spaces "
                f" not {type(env.action_space)}"
            )

        new_box, forward, _ = rescale_box(env.observation_space, min, max)

        self.env = env
        self.func = forward
        self.observation_space = new_box


class FlattenObservation[ActType, ObsType](
    AbstractPureObservationWrapper[Float[Array, " flat"], ActType, ObsType]
):
    """Flatten the observation space into a 1-D array."""

    env: AbstractEnvLike[ActType, ObsType]
    func: Callable[[ObsType], Float[Array, " flat"]]
    observation_space: Box

    def __init__(self, env: AbstractEnvLike[ActType, ObsType]):
        self.env = env
        self.func = self.env.observation_space.flatten_sample
        self.observation_space = Box(
            -jnp.inf,
            jnp.inf,
            shape=(int(jnp.asarray(self.env.observation_space.flat_size)),),
        )
