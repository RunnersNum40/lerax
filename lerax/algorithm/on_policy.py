from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import cast

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.env import AbstractEnvLike
from lerax.policy import (
    AbstractActorCriticPolicy,
    AbstractStatefulActorCriticPolicy,
    AbstractStatelessActorCriticPolicy,
    StatefulWrapper,
)
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm
from .utils import (
    EpisodeStats,
    IterationCarry,
    JITProgressBar,
    JITSummaryWriter,
    StepCarry,
)


class AbstractOnPolicyAlgorithm(AbstractAlgorithm):
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

    def step(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulActorCriticPolicy,
        carry: StepCarry,
        *,
        key: Key,
    ) -> tuple[StepCarry, RolloutBuffer]:
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            bootstrap_key,
            reset_key,
        ) = jr.split(key, 7)

        observation = env.observation(carry.env_state, key=observation_key)
        next_policy_state, action, value, log_prob = policy.action_and_value(
            carry.policy_state, observation, key=action_key
        )

        next_env_state = env.transition(carry.env_state, action, key=transition_key)

        reward = env.reward(carry.env_state, action, next_env_state, key=reward_key)
        termination = env.terminal(next_env_state, key=terminal_key)
        truncation = env.truncate(next_env_state)
        done = termination | truncation
        next_episode_stats = carry.episode_stats.next(reward, done)

        # Bootstrap reward if truncated
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
            done, lambda: env.initial(key=reset_key), lambda: next_env_state
        )

        # Reset policy state if done
        next_policy_state = lax.cond(
            done, lambda: policy.reset(), lambda: next_policy_state
        )

        return (
            StepCarry(next_env_state, next_policy_state, next_episode_stats),
            RolloutBuffer(
                observations=observation,
                actions=action,
                rewards=reward,
                dones=done,
                log_probs=log_prob,
                values=value,
                states=next_policy_state,
            ),
        )

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulActorCriticPolicy,
        carry: StepCarry,
        key: Key,
    ) -> tuple[StepCarry, RolloutBuffer, EpisodeStats]:
        key, observation_key = jr.split(key, 2)

        def scan_step(
            carry: StepCarry, key: Key
        ) -> tuple[StepCarry, tuple[RolloutBuffer, EpisodeStats]]:
            carry, rollout = self.step(env, policy, carry, key=key)
            return carry, (rollout, carry.episode_stats)

        carry, (rollout_buffer, episode_stats) = filter_scan(
            scan_step, carry, jr.split(key, self.num_steps)
        )

        _, next_value = policy.value(
            carry.policy_state, env.observation(carry.env_state, key=observation_key)
        )
        rollout_buffer = rollout_buffer.compute_returns_and_advantages(
            next_value, self.gae_lambda, self.gamma
        )
        return carry, rollout_buffer, episode_stats

    @abstractmethod
    def train[PolicyType: AbstractStatefulActorCriticPolicy](
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
        *,
        key: Key,
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        """Train the policy using the rollout buffer."""

    def iteration(
        self,
        env: AbstractEnvLike,
        carry: IterationCarry,
        *,
        key: Key,
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
    ) -> IterationCarry:
        rollout_key, train_key = jr.split(key, 2)
        if self.num_envs == 1:
            step_carry, rollout_buffer, episode_stats = self.collect_rollout(
                env, carry.policy, carry.step_carry, key=rollout_key
            )
        else:
            step_carry, rollout_buffer, episode_stats = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, 0, 0)
            )(env, carry.policy, carry.step_carry, jr.split(rollout_key, self.num_envs))

        policy, opt_state, log = self.train(
            carry.policy, carry.opt_state, rollout_buffer, key=train_key
        )

        if progress_bar is not None:
            progress_bar.update(advance=self.num_steps * self.num_envs)

        if tb_writer is not None:
            first_step = carry.iteration_count * self.num_steps * self.num_envs
            final_step = first_step + self.num_steps * self.num_envs - 1
            log["learning_rate"] = optax.tree_utils.tree_get(
                opt_state, "learning_rate", jnp.nan
            )
            tb_writer.add_dict(log, prefix="train", global_step=final_step)
            tb_writer.log_episode_stats(episode_stats, first_step=first_step)

        return IterationCarry(carry.iteration_count + 1, step_carry, policy, opt_state)

    def learn[PolicyType: AbstractActorCriticPolicy](
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        total_timesteps: int,
        *,
        key: Key,
        show_progress_bar: bool = False,
        tb_log: str | bool = False,
    ) -> PolicyType:
        if isinstance(policy, AbstractStatelessActorCriticPolicy):
            _policy = policy.into_stateful()
        elif isinstance(policy, AbstractStatefulActorCriticPolicy):
            _policy = policy
        else:
            raise TypeError("Unknown policy type.")

        init_key, learn_key = jr.split(key, 2)

        carry = self.init_iteration_carry(env, _policy, key=init_key)

        progress_bar = self.init_progress_bar(
            env, _policy, total_timesteps, show_progress_bar
        )
        tb_writer = self.init_tensorboard(env, _policy, tb_log)
        num_iterations = total_timesteps // (self.num_steps * self.num_envs)

        @eqx.filter_jit
        def learn(carry: IterationCarry) -> IterationCarry:
            def scan_iteration(carry: tuple[IterationCarry, Key], _):
                it_carry, key = carry
                iter_key, next_key = jr.split(key, 2)
                it_carry = self.iteration(
                    env,
                    it_carry,
                    key=iter_key,
                    progress_bar=progress_bar,
                    tb_writer=tb_writer,
                )
                return (it_carry, next_key), None

            (carry, _), _ = filter_scan(
                scan_iteration, (carry, learn_key), length=num_iterations
            )

            return carry

        carry = learn(carry)

        if progress_bar is not None:
            progress_bar.stop()

        if isinstance(policy, AbstractStatelessActorCriticPolicy):
            if isinstance(carry.policy, StatefulWrapper):
                return carry.policy.into_stateless()
            else:
                raise TypeError(
                    f"Unknown expected StatefulWrapper policy type. Got {type(carry.policy)}. This is likely a bug."
                )
        else:
            return cast(PolicyType, carry.policy)
