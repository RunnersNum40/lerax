import dataclasses

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from oryx.env import AbstractEnvLike
from oryx.spaces import AbstractSpace

from .base_wrapper import AbstractNoRenderOrCloseWrapper


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
        self.episode_length = jnp.asarray(episode_length)
        self.episode_reward = jnp.asarray(episode_reward)
        self.episode_done = jnp.asarray(episode_done)

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
