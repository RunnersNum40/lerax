from __future__ import annotations

from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState

from .base_wrapper import AbstractWrapper


class AbstractPureTransformRewardWrapper[
    StateType: AbstractEnvLikeState, ActType, ObsType
](AbstractWrapper[StateType, ActType, ObsType, StateType, ActType, ObsType]):
    """
    Apply a *pure* (stateless) function to every reward emitted by the wrapped
    environment.
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[Float[Array, ""]], Float[Array, ""]]]

    def initial(self, *, key: Key) -> StateType:
        return self.env.initial(key=key)

    def transition(self, state: StateType, action: ActType, *, key: Key) -> StateType:
        return self.env.transition(state, action, key=key)

    def observation(self, state: StateType, *, key: Key) -> ObsType:
        return self.env.observation(state, key=key)

    def reward(
        self, state: StateType, action: ActType, next_state: StateType, *, key: Key
    ) -> Float[Array, ""]:
        return self.func(self.env.reward(state, action, next_state, key=key))

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


class TransformReward[StateType: AbstractEnvLikeState, ActType, ObsType](
    AbstractPureTransformRewardWrapper[StateType, ActType, ObsType]
):
    """
    Apply an arbitrary function to the rewards emitted by the wrapped environment.

    Attributes:
        env: The environment to wrap.
        func: The function to apply to the rewards.

    Args:
        env: The environment to wrap.
        func: The function to apply to the rewards.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType]
    func: Callable[[Float[Array, ""]], Float[Array, ""]]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType],
        func: Callable[[Float[Array, ""]], Float[Array, ""]],
    ):
        self.env = env
        self.func = func


class ClipReward[StateType: AbstractEnvLikeState, ActType, ObsType](
    AbstractPureTransformRewardWrapper[StateType, ActType, ObsType]
):
    """
    Cip the rewards emitted by the wrapped environment to a specified range.

    Attributes:
        env: The environment to wrap.
        min: The minimum reward value.
        max: The maximum reward value.

    Args:
        env: The environment to wrap.
        min: The minimum reward value.
        max: The maximum reward value.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType]
    func: Callable[[Float[Array, ""]], Float[Array, ""]]
    min: Float[Array, ""]
    max: Float[Array, ""]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType],
        min: Float[ArrayLike, ""] = jnp.asarray(-1.0),
        max: Float[ArrayLike, ""] = jnp.asarray(1.0),
    ):
        self.env = env
        self.min = jnp.asarray(min)
        self.max = jnp.asarray(max)
        self.func = partial(jnp.clip, min=self.min, max=self.max)
