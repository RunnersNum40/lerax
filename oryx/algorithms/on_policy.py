from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key
from tensorboardX import SummaryWriter

from oryx.buffers import RolloutBuffer
from oryx.env import AbstractEnvLike
from oryx.policies import AbstractActorCriticPolicy

from .base_algorithm import AbstractAlgorithm


class AbstractOnPolicyAlgorithm[ActType, ObsType](AbstractAlgorithm[ActType, ObsType]):
    """Base class for on policy algorithms."""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    policy: eqx.AbstractVar[AbstractActorCriticPolicy[Float, ActType, ObsType]]

    def step(
        self,
        state: eqx.nn.State,
        previous: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> tuple[eqx.nn.State, RolloutBuffer[ActType, ObsType]]:
        """Perform a single step in the environment."""
        observation = previous.observations

        action_key, env_key, reset_key = jr.split(key, 3)

        policy_state = state.substate(self.policy)
        policy_state, action, value, log_prob = self.policy(
            policy_state, observation, key=action_key
        )
        state = state.update(policy_state)

        env_state = state.substate(self.env)
        env_state, observation, reward, termination, truncation, info = self.env.step(
            env_state, action, key=env_key
        )
        state = state.update(env_state)

        # TODO: Add TensorBoard logging

        def reset_env() -> tuple[eqx.nn.State, ObsType, dict]:
            env_state = state.substate(self.env)
            env_state, observation, info = self.env.reset(env_state, key=reset_key)
            _state = state.update(env_state)  # _ prefix to avoid shadowing `state`

            policy_state = state.substate(self.policy)
            policy_state = self.policy.reset(policy_state)
            _state = _state.update(policy_state)

            return _state, observation, info

        def identity() -> tuple[eqx.nn.State, ObsType, dict]:
            return state, observation, info

        done = previous.terminations | previous.truncations

        state, observation, info = lax.cond(
            done,
            reset_env,
            identity,
            state,
        )

        return state, RolloutBuffer(
            observations=observation,
            actions=action,
            rewards=reward,
            terminations=termination,
            truncations=truncation,
            log_probs=log_prob,
            values=value,
            states=state,
        )

    def collect_rollout(
        self,
        state: eqx.nn.State,
        num_steps: int,
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> tuple[eqx.nn.State, RolloutBuffer[ActType, ObsType]]:
        """Collect a rollout from the environment and store it in a buffer."""
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> AbstractOnPolicyAlgorithm:
        """Train the policy using the rollout buffer."""

    def learn(
        self,
        callback: Callable | None = None,
        *,
        key: Key | None = None,
        progress_bar: bool = False,
        tb_log_name: str | None = None,
        log_interval: int = 100,
    ) -> AbstractOnPolicyAlgorithm[ActType, ObsType]:
        """Return a trained model."""
        raise NotImplementedError
