from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

import equinox as eqx
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Int, Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import (
    AbstractActorCriticPolicy,
    AbstractPolicyState,
    AbstractStatefulActorCriticPolicy,
    AbstractStatelessActorCriticPolicy,
)
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm
from .utils import EpisodeStatisticsAccumulator, JITProgressBar, JITSummaryWriter


class StepCarry(eqx.Module):
    step_count: Int[Array, ""]
    last_obs: Array
    last_termination: Bool[Array, ""]
    last_truncation: Bool[Array, ""]

    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState


class IterationCarry(eqx.Module):
    step_carry: StepCarry
    policy: AbstractStatefulActorCriticPolicy
    opt_state: optax.OptState


class AbstractOnPolicyAlgorithm(AbstractAlgorithm):
    """Base class for on-policy algorithms."""

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    gae_lambda: eqx.AbstractVar[float]
    gamma: eqx.AbstractVar[float]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def step(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulActorCriticPolicy,
        carry: StepCarry,
        *,
        key: Key,
    ) -> tuple[StepCarry, RolloutBuffer, EpisodeStatisticsAccumulator | None, dict]:
        action_key, env_key, reset_key = jr.split(key, 3)

        policy_state, action, value, log_prob = policy.action_and_value(
            carry.policy_state, carry.last_obs, key=action_key
        )

        env_state, observation, reward, termination, truncation, info = env.step(
            carry.env_state, action, key=env_key
        )

        episode_stats = (
            EpisodeStatisticsAccumulator.from_episode_stats(info["episode"])
            if "episode" in info
            else None
        )

        def reset_env(env_state, policy_state):
            env_state, reset_observation, reset_info = env.reset(key=reset_key)
            policy_state = policy.reset()
            return env_state, policy_state, reset_observation, reset_info

        def identity(env_state, policy_state):
            return env_state, policy_state, observation, info

        done = jnp.logical_or(termination, truncation)
        env_state, policy_state, observation, info = lax.cond(
            done, reset_env, identity, env_state, policy_state
        )

        return (
            StepCarry(
                carry.step_count + 1,
                observation,
                termination,
                truncation,
                env_state,
                policy_state,
            ),
            RolloutBuffer(
                observations=carry.last_obs,
                actions=action,
                rewards=reward,
                terminations=carry.last_termination,
                truncations=carry.last_truncation,
                log_probs=log_prob,
                values=value,
                states=carry.policy_state,
            ),
            episode_stats,
            info,
        )

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulActorCriticPolicy,
        carry: StepCarry,
        *,
        key: Key,
    ) -> tuple[StepCarry, RolloutBuffer, EpisodeStatisticsAccumulator | None]:
        def scan_step(carry: tuple[StepCarry, Key], _):
            previous, key = carry
            step_key, carry_key = jr.split(key, 2)
            previous, rollout, episode_stats, _ = self.step(
                env, policy, previous, key=step_key
            )
            return (previous, carry_key), (rollout, episode_stats)

        (carry, _), (rollout_buffer, episode_stats) = filter_scan(
            scan_step, (carry, key), length=self.num_steps
        )

        next_done = jnp.logical_or(carry.last_termination, carry.last_truncation)
        _, next_value = policy.value(carry.policy_state, carry.last_obs)
        rollout_buffer = rollout_buffer.compute_returns_and_advantages(
            next_value, next_done, self.gae_lambda, self.gamma
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
        step_carry, rollout_buffer, episode_stats = self.collect_rollout(
            env, carry.policy, carry.step_carry, key=rollout_key
        )
        policy, opt_state = carry.policy, carry.opt_state
        policy, opt_state, log = self.train(
            policy, opt_state, rollout_buffer, key=train_key
        )

        if progress_bar is not None:
            progress_bar.update(advance=self.num_steps)
        if tb_writer is not None:
            log["learning_rate"] = optax.tree_utils.tree_get(
                opt_state, "learning_rate", jnp.nan
            )
            tb_writer.add_dict(
                log, prefix="train", global_step=carry.step_carry.step_count
            )
            if episode_stats is not None:
                tb_writer.log_episode_stats(
                    episode_stats, global_step=carry.step_carry.step_count
                )

        return IterationCarry(step_carry, policy, opt_state)

    def init_iteration_carry(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulActorCriticPolicy,
        *,
        key: Key,
    ) -> IterationCarry:
        env_state, next_obs, _ = env.reset(key=key)
        policy_state = policy.reset()
        opt_state = self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array))

        return IterationCarry(
            StepCarry(
                jnp.asarray(0),
                next_obs,
                jnp.asarray(False),
                jnp.asarray(False),
                env_state,
                policy_state,
            ),
            policy,
            opt_state,
        )

    def init_tensorboard(
        self,
        env: AbstractEnvLike,
        policy: AbstractActorCriticPolicy,
        tb_log: str | bool,
    ) -> JITSummaryWriter | None:
        if tb_log is False:
            return None

        if tb_log is True:
            tb_log = f"logs/{type(policy).__name__}_{env.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return JITSummaryWriter(tb_log)

    def init_progress_bar(
        self,
        env: AbstractEnvLike,
        policy: AbstractActorCriticPolicy,
        total_timesteps: int,
        show_progress_bar: bool,
    ) -> JITProgressBar | None:
        if show_progress_bar:
            name = f"Training {type(policy).__name__} on {env.name}"
            progress_bar = JITProgressBar(name, total=total_timesteps)
            progress_bar.start()
            return progress_bar
        else:
            return None

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
        num_iterations = total_timesteps // self.num_steps

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

        if progress_bar is not None:
            progress_bar.stop()

        if isinstance(policy, AbstractStatelessActorCriticPolicy):
            return carry.policy.into_stateless()
        else:
            return carry.policy
