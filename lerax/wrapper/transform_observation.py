from __future__ import annotations

from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvState
from lerax.space import AbstractSpace, Box

from .base_wrapper import AbstractWrapper
from .utils import rescale_box


class AbstractPureObservationWrapper[
    WrapperObsType, StateType: AbstractEnvState, ActType, ObsType
](AbstractWrapper[StateType, ActType, WrapperObsType, StateType, ActType, ObsType]):
    """
    Apply a pure function to every observation that leaves the environment.
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[ObsType], WrapperObsType]]
    observation_space: eqx.AbstractVar[AbstractSpace[WrapperObsType]]

    def reset(self, *, key: Key) -> tuple[StateType, WrapperObsType, dict]:
        state, observation, info = self.env.reset(key=key)
        return state, self.func(observation), info

    def step(self, state: StateType, action: ActType, *, key: Key):
        state, observation, reward, termination, truncation, info = self.env.step(
            state, action, key=key
        )
        return state, self.func(observation), reward, termination, truncation, info

    def render(self, state: StateType):
        self.env.render(state)

    def close(self):
        self.env.close()


class ClipObservation[StateType: AbstractEnvState](
    AbstractPureObservationWrapper[
        Float[Array, " ..."], StateType, Float[Array, " ..."], Float[Array, " ..."]
    ],
):
    """
    Clips every observation to the environment's observation space.
    """

    env: AbstractEnvLike
    func: Callable
    observation_space: Box

    def __init__(self, env: AbstractEnvLike):
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                "ClipObservation only supports `Box` observation spaces"
                f" not {type(env.observation_space)}"
            )

        self.env = env
        self.func = partial(
            jnp.clip,
            min=env.observation_space.low,
            max=env.observation_space.high,
        )
        self.observation_space = env.observation_space


class RescaleObservation[StateType: AbstractEnvState](
    AbstractPureObservationWrapper[
        Float[Array, " ..."], StateType, Float[Array, " ..."], Float[Array, " ..."]
    ],
):
    """Affinely rescale a box observation to a different range"""

    env: AbstractEnvLike
    func: Callable
    observation_space: Box

    def __init__(
        self,
        env: AbstractEnvLike,
        min: Float[Array, " ..."] = jnp.array(-1.0),
        max: Float[Array, " ..."] = jnp.array(1.0),
    ):
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                "RescaleObservation only supports `Box` observation spaces"
                f" not {type(env.action_space)}"
            )

        new_box, forward, _ = rescale_box(env.observation_space, min, max)

        self.env = env
        self.func = forward
        self.observation_space = new_box


class FlattenObservation[StateType: AbstractEnvState, ObsType](
    AbstractPureObservationWrapper[
        Float[Array, " flat"], StateType, Float[Array, " ..."], ObsType
    ]
):
    """Flatten the observation space into a 1-D array."""

    env: AbstractEnvLike
    func: Callable
    observation_space: Box

    def __init__(self, env: AbstractEnvLike):
        self.env = env
        self.func = self.env.observation_space.flatten_sample
        self.observation_space = Box(
            -jnp.inf,
            jnp.inf,
            shape=(int(jnp.asarray(self.env.observation_space.flat_size)),),
        )
