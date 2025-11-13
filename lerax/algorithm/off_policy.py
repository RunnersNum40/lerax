from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy, AbstractStatefulPolicy
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm
from .utils import (
    EpisodeStats,
    IterationCarry,
    JITProgressBar,
    JITSummaryWriter,
    StepCarry,
)


class AbstractOffPolicyAlgorithm[PolicyType: AbstractPolicy](
    AbstractAlgorithm[PolicyType]
):
    buffer_size: eqx.AbstractVar[int]
    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    gamma: eqx.AbstractVar[float]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def step(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulPolicy,
        carry: StepCarry,
        key: Key,
    ) -> tuple[StepCarry, ReplayBuffer]:
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            next_observation_key,
            reset_key,
        ) = jr.split(key, 7)

        observation = env.observation(carry.env_state, key=observation_key)
        next_policy_state, action = policy(
            carry.policy_state, observation, key=action_key
        )

        next_env_state = env.transition(carry.env_state, action, key=transition_key)

        reward = env.reward(carry.env_state, action, next_env_state, key=reward_key)
        termination = env.terminal(carry.env_state, key=terminal_key)
        truncation = env.truncate(carry.env_state)
        done = termination | truncation
        timeout = truncation & ~termination

        next_episode_stats = carry.episode_stats.next(reward, done)

        next_observation = env.observation(next_env_state, key=next_observation_key)

        next_env_state = lax.cond(
            done, lambda: env.initial(key=reset_key), lambda: next_env_state
        )

        next_policy_state = lax.cond(
            done, lambda: policy.reset(), lambda: next_policy_state
        )

        return StepCarry(
            next_env_state, next_policy_state, next_episode_stats
        ), ReplayBuffer(
            observations=observation,
            next_observations=next_observation,
            actions=action,
            rewards=reward,
            dones=done,
            timeouts=timeout,
            states=carry.policy_state,
        )

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulPolicy,
        carry: StepCarry,
        key: Key,
    ) -> tuple[StepCarry, ReplayBuffer, EpisodeStats]:
        def scan_step(
            carry: StepCarry, key: Key
        ) -> tuple[StepCarry, tuple[ReplayBuffer, EpisodeStats]]:
            carry, buffer = self.step(env, policy, carry, key)
            return carry, (buffer, carry.episode_stats)

        carry, (replay_buffer, episode_stats) = filter_scan(
            scan_step, carry, jr.split(key, self.num_steps)
        )

        return carry, replay_buffer, episode_stats

    @abstractmethod
    def train[WrapperPolicyType: AbstractStatefulPolicy](
        self,
        policy: WrapperPolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        *,
        key: Key,
    ) -> tuple[WrapperPolicyType, optax.OptState, dict[str, Scalar]]:
        """Trains the policy using data from the replay buffer."""

    def iteration[WrapperPolicyType: AbstractStatefulPolicy](
        self,
        env: AbstractEnvLike,
        carry: IterationCarry[WrapperPolicyType],
        *,
        key: Key,
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
    ) -> IterationCarry[WrapperPolicyType]:
        rollout_key, train_key = jr.split(key, 2)
        if self.num_steps == 1:
            step_carry, replay_buffer, episode_stats = self.collect_rollout(
                env, carry.policy, carry.step_carry, key=rollout_key
            )
        else:
            step_carry, replay_buffer, episode_stats = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, 0, 0)
            )(
                env,
                carry.policy,
                carry.step_carry,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, log = self.train(
            carry.policy, carry.opt_state, replay_buffer, key=train_key
        )

        if progress_bar is not None:
            progress_bar.update(self.num_envs * self.num_steps)

        if tb_writer is not None:
            first_step = carry.iteration_count * self.num_steps * self.num_envs
            final_step = first_step + self.num_steps * self.num_envs - 1
            log["learning_rate"] = optax.tree_utils.tree_get(
                opt_state, "learning_rate", jnp.nan
            )
            tb_writer.add_dict(log, prefix="train", global_step=final_step)
            tb_writer.log_episode_stats(episode_stats, first_step=first_step)

        return IterationCarry(carry.iteration_count + 1, step_carry, policy, opt_state)

    def learn(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        total_timesteps: int,
        *,
        key: Key,
        show_progress_bar: bool = False,
        tb_log: str | bool = False,
    ) -> PolicyType:
        raise NotImplementedError
