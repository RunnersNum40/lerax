from __future__ import annotations

from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Key

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

    def reset(self, *, key: Key) -> tuple[StateType, ObsType, dict]:
        return self.env.reset(key=key)

    def step(self, state: StateType, action: ActType, *, key: Key):
        state, observation, reward, termination, truncation, info = self.env.step(
            state, action, key=key
        )
        return state, observation, self.func(reward), termination, truncation, info

    def render(self, state: StateType):
        self.env.render(state)

    def close(self):
        self.env.close()


class ClipReward[StateType: AbstractEnvLikeState, ActType, ObsType](
    AbstractPureTransformRewardWrapper[StateType, ActType, ObsType]
):
    """
    Element-wise clip of rewards:  `reward -> clamp(min, max)`.
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
