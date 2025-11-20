from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Int, Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicyState, AbstractStatefulPolicy
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm, AbstractIterationCarry, AbstractStepCarry
from .utils import EpisodeStats, JITProgressBar, JITSummaryWriter


class StepCarry[PolicyType: AbstractStatefulPolicy](AbstractStepCarry):
    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    episode_stats: EpisodeStats
    buffer: ReplayBuffer

    @classmethod
    def initial(
        cls, size: int, env: AbstractEnvLike, policy: PolicyType, key: Key
    ) -> StepCarry[PolicyType]:
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        buffer = ReplayBuffer(
            size, env.observation_space, env.action_space, policy_state
        )
        return cls(env_state, policy_state, EpisodeStats.initial(), buffer)


class IterationCarry[PolicyType: AbstractStatefulPolicy](AbstractIterationCarry):
    iteration_count: Int[Array, ""]
    step_carry: StepCarry[PolicyType]
    policy: PolicyType
    opt_state: optax.OptState


class AbstractOffPolicyAlgorithm[PolicyType: AbstractStatefulPolicy](
    AbstractAlgorithm[PolicyType]
):
    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    buffer_size: eqx.AbstractVar[int]
    gamma: eqx.AbstractVar[float]
    learning_starts: eqx.AbstractVar[int]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    @abstractmethod
    def per_step(self, step_carry: StepCarry[PolicyType]) -> StepCarry[PolicyType]:
        """Process the step carry after each step."""

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        carry: StepCarry[PolicyType],
        key: Key,
    ) -> StepCarry[PolicyType]:
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            next_observation_key,
            env_reset_key,
            policy_reset_key,
        ) = jr.split(key, 8)

        observation = env.observation(carry.env_state, key=observation_key)
        policy_state, action = policy(carry.policy_state, observation, key=action_key)

        next_env_state = env.transition(carry.env_state, action, key=transition_key)

        reward = env.reward(carry.env_state, action, next_env_state, key=reward_key)
        termination = env.terminal(next_env_state, key=terminal_key)
        truncation = env.truncate(next_env_state)
        done = termination | truncation
        timeout = truncation & ~termination
        next_observation = env.observation(next_env_state, key=next_observation_key)

        next_episode_stats = carry.episode_stats.next(reward, done)

        next_env_state = lax.cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )

        next_policy_state = lax.cond(
            done, lambda: policy.reset(key=policy_reset_key), lambda: policy_state
        )

        replay_buffer = carry.buffer.add(
            observation,
            next_observation,
            action,
            reward,
            done,
            timeout,
            carry.policy_state,
            policy_state,
        )

        return StepCarry(
            next_env_state, next_policy_state, next_episode_stats, replay_buffer
        )

    def collect_learning_starts(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        carry: StepCarry[PolicyType],
        key: Key,
    ) -> StepCarry[PolicyType]:
        def scan_step(carry: StepCarry, key: Key) -> tuple[StepCarry, None]:
            carry = self.step(env, policy, carry, key)
            return carry, None

        carry, _ = filter_scan(scan_step, carry, jr.split(key, self.learning_starts))

        return carry

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        carry: StepCarry[PolicyType],
        key: Key,
    ) -> tuple[StepCarry[PolicyType], EpisodeStats]:
        def scan_step(carry: StepCarry, key: Key) -> tuple[StepCarry, EpisodeStats]:
            carry = self.step(env, policy, carry, key)
            return self.per_step(carry), carry.episode_stats

        carry, episode_stats = filter_scan(
            scan_step, carry, jr.split(key, self.num_steps)
        )

        return carry, episode_stats

    @abstractmethod
    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        *,
        key: Key,
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        """Trains the policy using data from the replay buffer."""

    def init_iteration_carry(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key,
    ) -> IterationCarry[PolicyType]:
        init_key, starts_key = jr.split(key, 2)
        if self.num_envs == 1:
            step_carry = StepCarry.initial(self.buffer_size, env, policy, init_key)
            step_carry = self.collect_learning_starts(
                env, policy, step_carry, key=starts_key
            )
        else:
            step_carry = jax.vmap(StepCarry.initial, in_axes=(None, None, None, 0))(
                self.buffer_size // self.num_envs,
                env,
                policy,
                jr.split(init_key, self.num_envs),
            )
            step_carry = jax.vmap(
                self.collect_learning_starts, in_axes=(None, None, 0, 0)
            )(env, policy, step_carry, jr.split(starts_key, self.num_envs))

        return IterationCarry(
            jnp.array(0, dtype=int),
            step_carry,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
        )

    @abstractmethod
    def per_iteration(
        self, iteration_carry: IterationCarry[PolicyType]
    ) -> IterationCarry[PolicyType]:
        """Process the iteration carry after each iteration."""

    def iteration(
        self,
        env: AbstractEnvLike,
        carry: IterationCarry[PolicyType],
        *,
        key: Key,
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
    ) -> IterationCarry[PolicyType]:
        rollout_key, train_key = jr.split(key, 2)
        if self.num_envs == 1:
            step_carry, episode_stats = self.collect_rollout(
                env, carry.policy, carry.step_carry, rollout_key
            )
        else:
            step_carry, episode_stats = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, 0, 0)
            )(
                env,
                carry.policy,
                carry.step_carry,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, log = self.train(
            carry.policy, carry.opt_state, step_carry.buffer, key=train_key
        )

        if progress_bar is not None:
            progress_bar.update(advance=self.num_envs * self.num_steps)

        if tb_writer is not None:
            first_step = carry.iteration_count * self.num_steps * self.num_envs
            final_step = first_step + self.num_steps * self.num_envs - 1
            log["learning_rate"] = optax.tree_utils.tree_get(
                opt_state, "learning_rate", jnp.nan
            )
            tb_writer.add_dict(log, prefix="train", global_step=final_step)
            tb_writer.log_episode_stats(episode_stats, first_step=first_step)

        return self.per_iteration(
            IterationCarry(carry.iteration_count + 1, step_carry, policy, opt_state)
        )
