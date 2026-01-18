from __future__ import annotations

from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState

from .base_wrapper import AbstractWrapper, AbstractWrapperState


class PureTransformRewardState[StateType: AbstractEnvLikeState](
    AbstractWrapperState[StateType]
):
    env_state: StateType


class AbstractPureTransformRewardWrapper[
    StateType: AbstractEnvLikeState, ActType, ObsType, MaskType
](
    AbstractWrapper[
        PureTransformRewardState[StateType],
        ActType,
        ObsType,
        MaskType,
        StateType,
        ActType,
        ObsType,
        MaskType,
    ]
):
    """
    Apply a *pure* (stateless) function to every reward emitted by the wrapped
    environment.
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType, MaskType]]
    func: eqx.AbstractVar[Callable[[Float[Array, ""]], Float[Array, ""]]]

    def initial(self, *, key: Key) -> PureTransformRewardState[StateType]:
        return PureTransformRewardState(self.env.initial(key=key))

    def action_mask(
        self, state: PureTransformRewardState[StateType], *, key: Key
    ) -> MaskType | None:
        return self.env.action_mask(state.env_state, key=key)

    def transition(
        self, state: PureTransformRewardState[StateType], action: ActType, *, key: Key
    ) -> PureTransformRewardState[StateType]:
        return PureTransformRewardState(
            self.env.transition(state.env_state, action, key=key)
        )

    def observation(
        self, state: PureTransformRewardState[StateType], *, key: Key
    ) -> ObsType:
        return self.env.observation(state.env_state, key=key)

    def reward(
        self,
        state: PureTransformRewardState[StateType],
        action: ActType,
        next_state: PureTransformRewardState[StateType],
        *,
        key: Key,
    ) -> Float[Array, ""]:
        return self.func(
            self.env.reward(state.env_state, action, next_state.env_state, key=key)
        )

    def terminal(
        self, state: PureTransformRewardState[StateType], *, key: Key
    ) -> Bool[Array, ""]:
        return self.env.terminal(state.env_state, key=key)

    def truncate(self, state: PureTransformRewardState[StateType]) -> Bool[Array, ""]:
        return self.env.truncate(state.env_state)

    def state_info(self, state: PureTransformRewardState[StateType]) -> dict:
        return self.env.state_info(state.env_state)

    def transition_info(
        self,
        state: PureTransformRewardState[StateType],
        action: ActType,
        next_state: PureTransformRewardState[StateType],
    ) -> dict:
        return self.env.transition_info(state.env_state, action, next_state.env_state)


class TransformReward[StateType: AbstractEnvLikeState, ActType, ObsType, MaskType](
    AbstractPureTransformRewardWrapper[StateType, ActType, ObsType, MaskType]
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

    env: AbstractEnvLike[StateType, ActType, ObsType, MaskType]
    func: Callable[[Float[Array, ""]], Float[Array, ""]]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType, MaskType],
        func: Callable[[Float[Array, ""]], Float[Array, ""]],
    ):
        self.env = env
        self.func = func


class ClipReward[StateType: AbstractEnvLikeState, ActType, ObsType, MaskType](
    AbstractPureTransformRewardWrapper[StateType, ActType, ObsType, MaskType]
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

    env: AbstractEnvLike[StateType, ActType, ObsType, MaskType]
    func: Callable[[Float[Array, ""]], Float[Array, ""]]
    min: Float[Array, ""]
    max: Float[Array, ""]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType, MaskType],
        min: Float[ArrayLike, ""] = jnp.asarray(-1.0),
        max: Float[ArrayLike, ""] = jnp.asarray(1.0),
    ):
        self.env = env
        self.min = jnp.asarray(min)
        self.max = jnp.asarray(max)
        self.func = partial(jnp.clip, min=self.min, max=self.max)
