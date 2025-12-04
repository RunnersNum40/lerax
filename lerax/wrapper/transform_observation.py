from __future__ import annotations

from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Box

from .base_wrapper import AbstractWrapper
from .utils import rescale_box


class AbstractPureObservationWrapper[
    WrapperObsType, StateType: AbstractEnvLikeState, ActType, ObsType
](AbstractWrapper[StateType, ActType, WrapperObsType, StateType, ActType, ObsType]):
    """
    Apply a pure function to every observation that leaves the environment.
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[ObsType], WrapperObsType]]
    observation_space: eqx.AbstractVar[AbstractSpace[WrapperObsType]]

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    def initial(self, *, key: Key) -> StateType:
        return self.env.initial(key=key)

    def transition(self, state: StateType, action: ActType, *, key: Key) -> StateType:
        return self.env.transition(state, action, key=key)

    def observation(self, state: StateType, *, key: Key) -> WrapperObsType:
        return self.func(self.env.observation(state, key=key))

    def reward(
        self, state: StateType, action: ActType, next_state: StateType, *, key: Key
    ) -> Float[Array, ""]:
        return self.env.reward(state, action, next_state, key=key)

    def terminal(self, state: StateType, *, key: Key) -> Bool[Array, ""]:
        return self.env.terminal(state, key=key)

    def truncate(self, state: StateType) -> Bool[Array, ""]:
        return self.env.truncate(state)

    def state_info(self, state: StateType) -> dict:
        return self.env.state_info(state)

    def transition_info(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> dict:
        return self.env.transition_info(state, action, next_state)


class TransformObservation[
    WrapperObsType, StateType: AbstractEnvLikeState, ActType, ObsType
](AbstractPureObservationWrapper[WrapperObsType, StateType, ActType, ObsType]):
    """
    Apply an arbitrary function to every observation that leaves the environment.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType]
    func: Callable[[ObsType], WrapperObsType]
    observation_space: AbstractSpace[WrapperObsType]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType],
        func: Callable[[ObsType], WrapperObsType],
        observation_space: AbstractSpace[WrapperObsType],
    ):
        self.env = env
        self.func = func
        self.observation_space = observation_space


class ClipObservation[StateType: AbstractEnvLikeState](
    AbstractPureObservationWrapper[
        Float[Array, " ..."], StateType, Float[Array, " ..."], Float[Array, " ..."]
    ],
):
    """
    Clips every observation to the environment's observation space.

    Note:
        Only works with `Box` observation spaces.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.

    Raises:
        ValueError: If the environment's observation space is not a `Box`.
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


class RescaleObservation[StateType: AbstractEnvLikeState](
    AbstractPureObservationWrapper[
        Float[Array, " ..."], StateType, Float[Array, " ..."], Float[Array, " ..."]
    ],
):
    """
    Affine rescaling of the box observation space to a specified range.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.
        min: The minimum value of the rescaled observation space.
        max: The maximum value of the rescaled observation space.

    Raises:
        ValueError: If the environment's observation space is not a `Box`.
    """

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


class FlattenObservation[StateType: AbstractEnvLikeState, ObsType](
    AbstractPureObservationWrapper[
        Float[Array, " flat"], StateType, Float[Array, " ..."], ObsType
    ]
):
    """
    Flatten the observation space into a 1-D array.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.
    """

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
