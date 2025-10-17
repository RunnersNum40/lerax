from __future__ import annotations

import dataclasses

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
    env: AbstractEnvLike[StateType, ActType, ObsType]

    def __init__(self, env: AbstractEnvLike[StateType, ActType, ObsType]):
        self.env = env

    def reset(self, *, key: Key) -> tuple[StateType, ObsType, dict]:
        return self.env.reset(key=key)

    def step(
        self, state: StateType, action: ActType, *, key: Key
    ) -> tuple[
        StateType, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        return self.env.step(state, action, key=key)


class EpisodeStatisticsState[StateType: AbstractEnvLikeState](AbstractWrapperState):
    env_state: StateType

    episode_length: Int[Array, ""]
    episode_reward: Float[Array, ""]
    episode_done: Bool[Array, ""]

    def __init__(
        self,
        env_state: StateType,
        episode_length: Int[ArrayLike, ""] = 0,
        episode_reward: Float[ArrayLike, ""] = 0.0,
        episode_done: Bool[ArrayLike, ""] = False,
    ):
        self.env_state = env_state

        # Types must be specified to avoid weak types
        self.episode_length = jnp.array(episode_length, dtype=int)
        self.episode_reward = jnp.array(episode_reward, dtype=float)
        self.episode_done = jnp.array(episode_done, dtype=bool)

    def update(
        self,
        env_state: StateType,
        reward: Float[ArrayLike, ""],
        done: Bool[ArrayLike, ""],
    ):
        return dataclasses.replace(
            self,
            env_state=env_state,
            episode_length=self.episode_length + 1,
            episode_reward=self.episode_reward + reward,
            episode_done=jnp.array(done, dtype=bool),
        )

    def info(self) -> dict:
        return {
            "length": self.episode_length,
            "reward": self.episode_reward,
            "done": self.episode_done,
        }


class EpisodeStatistics[StateType: AbstractEnvLikeState, ActType, ObsType](
    AbstractWrapper[
        EpisodeStatisticsState[StateType], ActType, ObsType, StateType, ActType, ObsType
    ]
):
    env: AbstractEnvLike[StateType, ActType, ObsType]

    def __init__(self, env: AbstractEnvLike[StateType, ActType, ObsType]):
        self.env = env

    def reset(
        self, *, key: Key
    ) -> tuple[EpisodeStatisticsState[StateType], ObsType, dict]:
        env_state, obs, info = self.env.reset(key=key)
        state = EpisodeStatisticsState(env_state)
        info["episode"] = state.info()

        return state, obs, info

    def step(
        self, state: EpisodeStatisticsState[StateType], action: ActType, *, key: Key
    ) -> tuple[
        EpisodeStatisticsState[StateType],
        ObsType,
        Float[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        dict,
    ]:
        env_state, obs, reward, termination, truncation, info = self.env.step(
            state.env_state, action, key=key
        )
        state = state.update(env_state, reward, jnp.logical_or(termination, truncation))
        info["episode"] = state.info()

        return state, obs, reward, termination, truncation, info

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space

    def render(self, state: EpisodeStatisticsState[StateType]):
        self.env.render(state.env_state)

    def close(self):
        self.env.close()


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

    def reset(self, *, key: Key) -> tuple[TimeLimitState[StateType], ObsType, dict]:
        env_state, obs, info = self.env.reset(key=key)
        state = TimeLimitState(jnp.array(0, dtype=int), env_state)
        return state, obs, info

    def step(
        self, state: TimeLimitState[StateType], action: ActType, *, key: Key
    ) -> tuple[
        TimeLimitState[StateType],
        ObsType,
        Float[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        dict,
    ]:
        env_state, obs, reward, termination, truncation, info = self.env.step(
            state.env_state, action, key=key
        )

        step_count = state.step_count + 1
        truncation = jnp.logical_or(truncation, step_count >= self.max_episode_steps)

        state = TimeLimitState(step_count, env_state)

        return state, obs, reward, termination, truncation, info

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space

    def render(self, state: TimeLimitState[StateType]):
        self.env.render(state.env_state)

    def close(self):
        self.env.close()


class AutoClose[StateType: AbstractEnvLikeState, ActType, ObsType]():
    """
    Closes the environment automatically when it is deleted.

    Not JIT-compatible.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType]

    def __init__(self, env: AbstractEnvLike[StateType, ActType, ObsType]):
        self.env = env

    def reset(self, *, key: Key) -> tuple[StateType, ObsType, dict]:
        return self.env.reset(key=key)

    def step(
        self, state: StateType, action: ActType, *, key: Key
    ) -> tuple[
        StateType, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        return self.env.step(state, action, key=key)

    def render(self, state: StateType):
        self.env.render(state)

    def close(self):
        self.env.close()

    def __del__(self):
        self.close()
