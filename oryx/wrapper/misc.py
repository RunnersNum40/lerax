from __future__ import annotations

import dataclasses

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from oryx.env import AbstractEnvLike
from oryx.space import AbstractSpace

from .base_wrapper import (
    AbstractNoActionOrObservationSpaceWrapper,
    AbstractNoRenderOrCloseWrapper,
)


class Identity[ActType, ObsType](
    AbstractNoRenderOrCloseWrapper[ActType, ObsType, ActType, ObsType],
    AbstractNoActionOrObservationSpaceWrapper[ActType, ObsType],
):
    env: AbstractEnvLike[ActType, ObsType]

    def __init__(self, env: AbstractEnvLike[ActType, ObsType]):
        self.env = env

    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        return self.env.reset(state, key=key)

    def step(
        self, state: eqx.nn.State, action: ActType, *, key: Key
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        return self.env.step(state, action, key=key)


class LogState(eqx.Module):
    episode_length: Int[Array, ""]
    episode_reward: Float[Array, ""]
    episode_done: Bool[Array, ""]

    def __init__(
        self,
        episode_length: Int[ArrayLike, ""] = 0,
        episode_reward: Float[ArrayLike, ""] = 0.0,
        episode_done: Bool[ArrayLike, ""] = False,
    ):
        # Types must be specified to avoid weak types
        self.episode_length = jnp.array(episode_length, dtype=int)
        self.episode_reward = jnp.array(episode_reward, dtype=float)
        self.episode_done = jnp.array(episode_done, dtype=bool)

    def update(self, reward: Float[ArrayLike, ""], done: Bool[ArrayLike, ""]):
        return dataclasses.replace(
            self,
            episode_length=self.episode_length + 1,
            episode_reward=self.episode_reward + reward,
            episode_done=jnp.asarray(done),
        )


class EpisodeStatistics[ActType, ObsType](
    AbstractNoRenderOrCloseWrapper[ActType, ObsType, ActType, ObsType]
):
    state_index: eqx.nn.StateIndex[LogState]
    env: AbstractEnvLike[ActType, ObsType]

    def __init__(self, env: AbstractEnvLike[ActType, ObsType]):
        self.env = env
        self.state_index = eqx.nn.StateIndex(LogState())

    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        env_state = state.substate(self.env)
        env_state, obs, info = self.env.reset(env_state, key=key)
        state = state.update(env_state)

        log_state = LogState()
        wrapper_state = state.substate(self.state_index)
        wrapper_state = wrapper_state.set(self.state_index, log_state)
        state = state.update(wrapper_state)

        info["episode"] = {
            "length": log_state.episode_length,
            "reward": log_state.episode_reward,
            "done": log_state.episode_done,
        }

        return state, obs, info

    def step(
        self, state: eqx.nn.State, action: ActType, *, key: Key
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        env_state = state.substate(self.env)
        env_state, obs, reward, termination, truncation, info = self.env.step(
            env_state, action, key=key
        )
        state = state.update(env_state)

        log_state = state.get(self.state_index)
        log_state = log_state.update(reward, jnp.logical_or(termination, truncation))
        state = state.set(self.state_index, log_state)

        info["episode"] = {
            "length": log_state.episode_length,
            "reward": log_state.episode_reward,
            "done": log_state.episode_done,
        }

        return state, obs, reward, termination, truncation, info

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space


class TimeLimit[ActType, ObsType](
    AbstractNoActionOrObservationSpaceWrapper[ActType, ObsType],
    AbstractNoRenderOrCloseWrapper[ActType, ObsType, ActType, ObsType],
):

    state_index: eqx.nn.StateIndex[Int[Array, ""]]
    env: AbstractEnvLike[ActType, ObsType]
    max_episode_steps: Int[Array, ""]

    def __init__(self, env: AbstractEnvLike[ActType, ObsType], max_episode_steps: int):
        self.env = env
        self.state_index = eqx.nn.StateIndex(jnp.array(0))
        self.max_episode_steps = jnp.asarray(max_episode_steps)

    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        env_state = state.substate(self.env)
        env_state, obs, info = self.env.reset(env_state, key=key)
        state = state.update(env_state)

        state = state.set(self.state_index, jnp.array(0))

        return state, obs, info

    def step(
        self, state: eqx.nn.State, action: ActType, *, key: Key
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        env_state = state.substate(self.env)
        env_state, obs, reward, termination, truncation, info = self.env.step(
            env_state, action, key=key
        )
        state = state.update(env_state)

        step_count = state.get(self.state_index) + 1
        state = state.set(self.state_index, step_count)

        truncation = jnp.logical_or(truncation, step_count >= self.max_episode_steps)

        return state, obs, reward, termination, truncation, info


class AutoClose[ActType, ObsType](
    AbstractNoActionOrObservationSpaceWrapper[ActType, ObsType],
    AbstractNoRenderOrCloseWrapper[ActType, ObsType, ActType, ObsType],
):
    """
    Closes the environment automatically when it is deleted.
    """

    env: AbstractEnvLike[ActType, ObsType]

    def __init__(self, env: AbstractEnvLike[ActType, ObsType]):
        self.env = env

    def reset(
        self, state: eqx.nn.State, *, key: Key
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        return self.env.reset(state, key=key)

    def step(
        self, state: eqx.nn.State, action: ActType, *, key: Key
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        return self.env.step(state, action, key=key)

    def close(self):
        self.env.close()

    def __del__(self):
        self.close()
