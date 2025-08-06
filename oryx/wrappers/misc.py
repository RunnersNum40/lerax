import dataclasses

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from oryx.env import AbstractEnvLike
from oryx.spaces import AbstractSpace

from .base_wrapper import AbstractNoRenderOrCloseWrapper


class LogState(eqx.Module):
    """A simple class to log the state of the environment."""

    episode_length: Int[Array, ""] = jnp.array(0)
    episode_reward: Float[Array, ""] = jnp.array(0.0)
    episode_done: Bool[Array, ""] = jnp.array(False)

    def update(self, reward: Float[ArrayLike, ""], done: Bool[ArrayLike, ""]):
        return dataclasses.replace(
            self,
            episode_length=self.episode_length + 1,
            episode_reward=self.episode_reward + reward,
            episode_done=jnp.asarray(done),
        )


class EpisodeStatisticsWrapper[ActType, ObsType](
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
        substate = state.substate(self.env)
        substate, obs, info = self.env.reset(substate, key=key)
        state = state.update(substate)

        log_state = LogState()
        state = state.set(self.state_index, log_state)

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
        substate = state.substate(self.env)
        substate, obs, reward, termination, truncation, info = self.env.step(
            substate, action, key=key
        )
        state = state.update(substate)

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
