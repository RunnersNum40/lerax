from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace

from .base_wrapper import (
    AbstractWrapper,
    AbstractWrapperState,
)


class Identity[StateType: AbstractEnvLikeState, ActType, ObsType](
    AbstractWrapper[StateType, ActType, ObsType, StateType, ActType, ObsType]
):
    """An identity wrapper that does nothing."""

    env: AbstractEnvLike[StateType, ActType, ObsType]

    def __init__(self, env: AbstractEnvLike[StateType, ActType, ObsType]):
        self.env = env

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space

    def initial(self, *, key: Key) -> StateType:
        return self.env.initial(key=key)

    def transition(self, state: StateType, action: ActType, *, key: Key) -> StateType:
        return self.env.transition(state, action, key=key)

    def observation(self, state: StateType, *, key: Key) -> ObsType:
        return self.env.observation(state, key=key)

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


class TimeLimitState[StateType: AbstractEnvLikeState](AbstractWrapperState):
    env_state: StateType
    step_count: Int[Array, ""]

    def __init__(self, step_count: Int[ArrayLike, ""], env_state: StateType):
        self.step_count = jnp.array(step_count, dtype=int)
        self.env_state = env_state


class TimeLimit[StateType: AbstractEnvLikeState, ActType, ObsType](
    AbstractWrapper[
        TimeLimitState[StateType], ActType, ObsType, StateType, ActType, ObsType
    ]
):

    env: AbstractEnvLike[StateType, ActType, ObsType]
    max_episode_steps: Int[Array, ""]

    def __init__(
        self, env: AbstractEnvLike[StateType, ActType, ObsType], max_episode_steps: int
    ):
        self.env = env
        self.max_episode_steps = jnp.array(max_episode_steps, dtype=int)

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
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
