from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace

from .base_wrapper import (
    AbstractWrapper,
    AbstractWrapperState,
)


class IdentityState[StateType: AbstractEnvLikeState](AbstractWrapperState[StateType]):
    env_state: StateType


class Identity[StateType: AbstractEnvLikeState, ActType, ObsType, MaskType](
    AbstractWrapper[
        IdentityState[StateType],
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
    An wrapper that does nothing.

    Attributes:
        env: The environment to wrap.

    Args:
        env: The environment to wrap.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType, MaskType]

    def __init__(self, env: AbstractEnvLike[StateType, ActType, ObsType, MaskType]):
        self.env = env

    @property
    def action_space(self) -> AbstractSpace[ActType, MaskType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType, Any]:
        return self.env.observation_space

    def initial(self, *, key: Key) -> IdentityState[StateType]:
        return IdentityState(self.env.initial(key=key))

    def action_mask(
        self, state: IdentityState[StateType], *, key: Key
    ) -> MaskType | None:
        return self.env.action_mask(state.env_state, key=key)

    def transition(
        self, state: IdentityState[StateType], action: ActType, *, key: Key
    ) -> IdentityState[StateType]:
        return IdentityState(self.env.transition(state.env_state, action, key=key))

    def observation(self, state: IdentityState[StateType], *, key: Key) -> ObsType:
        return self.env.observation(state.env_state, key=key)

    def reward(
        self,
        state: IdentityState[StateType],
        action: ActType,
        next_state: IdentityState[StateType],
        *,
        key: Key,
    ) -> Float[Array, ""]:
        return self.env.reward(state.env_state, action, next_state.env_state, key=key)

    def terminal(self, state: IdentityState[StateType], *, key: Key) -> Bool[Array, ""]:
        return self.env.terminal(state.env_state, key=key)

    def truncate(self, state: IdentityState[StateType]) -> Bool[Array, ""]:
        return self.env.truncate(state.env_state)

    def state_info(self, state: IdentityState[StateType]) -> dict:
        return self.env.state_info(state.env_state)

    def transition_info(
        self,
        state: IdentityState[StateType],
        action: ActType,
        next_state: IdentityState[StateType],
    ) -> dict:
        return self.env.transition_info(state.env_state, action, next_state.env_state)


class TimeLimitState[StateType: AbstractEnvLikeState](AbstractWrapperState):
    env_state: StateType
    step_count: Int[Array, ""]

    def __init__(self, step_count: Int[ArrayLike, ""], env_state: StateType):
        self.step_count = jnp.array(step_count, dtype=int)
        self.env_state = env_state


class TimeLimit[StateType: AbstractEnvLikeState, ActType, ObsType, MaskType](
    AbstractWrapper[
        TimeLimitState[StateType],
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
    Time limit wrapper that truncates episodes after a fixed number of steps.

    Attributes:
        env: The environment to wrap.
        max_episode_steps: The maximum number of steps per episode.

    Args:
        env: The environment to wrap.
        max_episode_steps: The maximum number of steps per episode.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType, MaskType]
    max_episode_steps: Int[Array, ""]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType, MaskType],
        max_episode_steps: int,
    ):
        self.env = env
        self.max_episode_steps = jnp.array(max_episode_steps, dtype=int)

    @property
    def action_space(self) -> AbstractSpace[ActType, MaskType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType, Any]:
        return self.env.observation_space

    def initial(self, *, key: Key) -> TimeLimitState[StateType]:
        env_state = self.env.initial(key=key)
        return TimeLimitState(step_count=0, env_state=env_state)

    def transition(
        self, state: TimeLimitState[StateType], action: ActType, *, key: Key
    ) -> TimeLimitState[StateType]:
        env_next_state = self.env.transition(state.env_state, action, key=key)
        return TimeLimitState(step_count=state.step_count + 1, env_state=env_next_state)

    def observation(self, state: TimeLimitState[StateType], *, key: Key) -> ObsType:
        return self.env.observation(state.env_state, key=key)

    def reward(
        self,
        state: TimeLimitState[StateType],
        action: ActType,
        next_state: TimeLimitState[StateType],
        *,
        key: Key,
    ) -> Float[Array, ""]:
        return self.env.reward(state.env_state, action, next_state.env_state, key=key)

    def terminal(
        self, state: TimeLimitState[StateType], *, key: Key
    ) -> Bool[Array, ""]:
        return self.env.terminal(state.env_state, key=key)

    def truncate(self, state: TimeLimitState[StateType]) -> Bool[Array, ""]:
        env_truncate = self.env.truncate(state.env_state)
        return env_truncate | (state.step_count >= self.max_episode_steps)

    def state_info(self, state: TimeLimitState[StateType]) -> dict:
        return self.env.state_info(state.env_state)

    def transition_info(
        self,
        state: TimeLimitState[StateType],
        action: ActType,
        next_state: TimeLimitState[StateType],
    ) -> dict:
        return self.env.transition_info(state.env_state, action, next_state.env_state)
