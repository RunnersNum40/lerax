from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Int, Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import (
    AbstractPolicyState,
    AbstractStatefulActorCriticPolicy,
    AbstractStatefulPolicy,
)

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState
from .utils import EpisodeStats, JITProgressBar, JITSummaryWriter


class OnPolicyStepState[PolicyType: AbstractStatefulPolicy](AbstractStepState):
    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    episode_stats: EpisodeStats

    @classmethod
    def initial(
        cls, env: AbstractEnvLike, policy: PolicyType, key: Key
    ) -> OnPolicyStepState[PolicyType]:
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        return cls(env_state, policy_state, EpisodeStats.initial())


class OnPolicyState[PolicyType: AbstractStatefulPolicy](
    AbstractAlgorithmState[PolicyType]
):
    iteration_count: Int[Array, ""]
    step_state: OnPolicyStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    tb_writer: JITSummaryWriter | None
    progress_bar: JITProgressBar | None


class AbstractOnPolicyAlgorithm[PolicyType: AbstractStatefulActorCriticPolicy](
    AbstractAlgorithm[PolicyType, OnPolicyState[PolicyType]]
):
    """
    Base class for on-policy algorithms.

    Generates rollouts using the current policy and estimates advantages and
    returns using GAE. Trains the policy using the collected rollouts.
    """

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    gae_lambda: eqx.AbstractVar[float]
    gamma: eqx.AbstractVar[float]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    @abstractmethod
    def per_step(
        self, step_state: OnPolicyStepState[PolicyType]
    ) -> OnPolicyStepState[PolicyType]:
        """Process the step carry after each step."""

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: OnPolicyStepState,
        *,
        key: Key,
    ) -> tuple[OnPolicyStepState, RolloutBuffer]:
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            bootstrap_key,
            env_reset_key,
            policy_reset_key,
        ) = jr.split(key, 8)

        observation = env.observation(state.env_state, key=observation_key)
        next_policy_state, action, value, log_prob = policy.action_and_value(
            state.policy_state, observation, key=action_key
        )

        next_env_state = env.transition(state.env_state, action, key=transition_key)

        reward = env.reward(state.env_state, action, next_env_state, key=reward_key)
        termination = env.terminal(next_env_state, key=terminal_key)
        truncation = env.truncate(next_env_state)
        done = termination | truncation
        next_episode_stats = state.episode_stats.next(reward, done)

        # Bootstrap reward if truncated
        # TODO: Check if a non-branched approach is faster
        reward = lax.cond(
            truncation,
            lambda: reward
            + self.gamma
            * policy.value(
                next_policy_state, env.observation(next_env_state, key=bootstrap_key)
            )[1],
            lambda: reward,
        )

        # Reset environment if done
        next_env_state = lax.cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )

        # Reset policy state if done
        next_policy_state = lax.cond(
            done, lambda: policy.reset(key=policy_reset_key), lambda: next_policy_state
        )

        return (
            OnPolicyStepState(next_env_state, next_policy_state, next_episode_stats),
            RolloutBuffer(
                observations=observation,
                actions=action,
                rewards=reward,
                dones=done,
                log_probs=log_prob,
                values=value,
                states=state.policy_state,
            ),
        )

    @abstractmethod
    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: OnPolicyStepState[PolicyType],
        key: Key,
    ) -> tuple[OnPolicyStepState[PolicyType], RolloutBuffer, EpisodeStats]:
        """Collect a rollout using the current policy."""

    @abstractmethod
    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: RolloutBuffer,
        *,
        key: Key,
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        """Train the policy using the rollout buffer."""

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key,
        tb_writer: JITSummaryWriter | None,
        progress_bar: JITProgressBar | None,
    ) -> OnPolicyState[PolicyType]:
        if self.num_envs == 1:
            step_state = OnPolicyStepState.initial(env, policy, key)
        else:
            step_state = jax.vmap(OnPolicyStepState.initial, in_axes=(None, None, 0))(
                env, policy, jr.split(key, self.num_envs)
            )

        return OnPolicyState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            tb_writer,
            progress_bar,
        )

    def iteration(
        self,
        state: OnPolicyState[PolicyType],
        *,
        key: Key,
    ) -> OnPolicyState[PolicyType]:
        rollout_key, train_key = jr.split(key, 2)
        if self.num_envs == 1:
            step_state, rollout_buffer, episode_stats = self.collect_rollout(
                state.env, state.policy, state.step_state, key=rollout_key
            )
        else:
            step_state, rollout_buffer, episode_stats = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, 0, 0)
            )(
                state.env,
                state.policy,
                state.step_state,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, log = self.train(
            state.policy, state.opt_state, rollout_buffer, key=train_key
        )

        state = state.next(step_state, policy, opt_state)
        self.write_tensorboard(state, log, episode_stats)
        self.update_progress_bar(state)

        return self.per_iteration(state)
