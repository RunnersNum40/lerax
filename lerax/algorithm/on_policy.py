from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

import equinox as eqx
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Int, Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.env import AbstractEnvLike
from lerax.policy import AbstractActorCriticPolicy
from lerax.utils import clone_state, filter_scan

from .base_algorithm import AbstractAlgorithm
from .utils import EpisodeStatisticsAccumulator, JITProgressBar, JITSummaryWriter


class StepCarry[ObsType](eqx.Module):
    step_count: Int[Array, ""]
    last_obs: ObsType
    last_termination: Bool[Array, ""]
    last_truncation: Bool[Array, ""]


class IterationCarry[ActType, ObsType](eqx.Module):
    step_carry: StepCarry[ObsType]
    policy: AbstractActorCriticPolicy[Float, ActType, ObsType]
    opt_state: optax.OptState


class AbstractOnPolicyAlgorithm[ActType, ObsType](AbstractAlgorithm[ActType, ObsType]):
    """Base class for on-policy algorithms."""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    gae_lambda: eqx.AbstractVar[float]
    gamma: eqx.AbstractVar[float]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def step(
        self,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        state: eqx.nn.State,
        carry: StepCarry[ObsType],
        *,
        key: Key,
    ) -> tuple[
        eqx.nn.State,
        StepCarry[ObsType],
        RolloutBuffer[ActType, ObsType],
        EpisodeStatisticsAccumulator | None,
        dict,
    ]:
        action_key, env_key, reset_key = jr.split(key, 3)

        policy_state = state.substate(policy)
        policy_state, action, value, log_prob = policy(
            policy_state, carry.last_obs, key=action_key
        )
        state = state.update(policy_state)

        env_state = state.substate(self.env)
        env_state, observation, reward, termination, truncation, info = self.env.step(
            env_state, action, key=env_key
        )
        state = state.update(env_state)

        episode_stats = (
            EpisodeStatisticsAccumulator.from_episode_stats(info["episode"])
            if "episode" in info
            else None
        )

        def reset_env(state: eqx.nn.State):
            env_state = state.substate(self.env)
            env_state, reset_observation, reset_info = self.env.reset(
                env_state, key=reset_key
            )
            state = state.update(env_state)

            pol_state = state.substate(policy)
            pol_state = policy.reset(pol_state)
            state = state.update(pol_state)
            return state, reset_observation, reset_info

        def identity(state: eqx.nn.State):
            return state, observation, info

        done = jnp.logical_or(termination, truncation)
        state, observation, info = lax.cond(done, reset_env, identity, state)

        return (
            state,
            StepCarry(carry.step_count + 1, observation, termination, truncation),
            RolloutBuffer(
                observations=carry.last_obs,
                actions=action,
                rewards=reward,
                terminations=carry.last_termination,
                truncations=carry.last_truncation,
                log_probs=log_prob,
                values=value,
                states=state.substate(policy),
            ),
            episode_stats,
            info,
        )

    def collect_rollout(
        self,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        state: eqx.nn.State,
        carry: StepCarry[ObsType],
        *,
        key: Key,
    ) -> tuple[
        eqx.nn.State,
        StepCarry[ObsType],
        RolloutBuffer[ActType, ObsType],
        EpisodeStatisticsAccumulator | None,
    ]:
        def scan_step(
            carry: tuple[eqx.nn.State, StepCarry[ObsType], Key],
            _,
        ):
            state, previous, key = carry
            step_key, carry_key = jr.split(key, 2)
            state, previous, rollout, episode_stats, _ = self.step(
                policy, state, previous, key=step_key
            )
            return (state, previous, carry_key), (rollout, episode_stats)

        (state, carry, _), (rollout_buffer, episode_stats) = filter_scan(
            scan_step, (state, carry, key), length=self.num_steps
        )

        next_done = jnp.logical_or(carry.last_termination, carry.last_truncation)
        _, next_value = policy.value(clone_state(state), carry.last_obs)
        rollout_buffer = rollout_buffer.compute_returns_and_advantages(
            next_value, next_done, self.gae_lambda, self.gamma
        )
        return state, carry, rollout_buffer, episode_stats

    @abstractmethod
    def train(
        self,
        state: eqx.nn.State,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
    ) -> tuple[
        eqx.nn.State,
        AbstractActorCriticPolicy[Float, ActType, ObsType],
        optax.OptState,
        dict[str, Scalar],
    ]:
        """Train the policy using the rollout buffer."""

    def initialize_iteration_carry(
        self,
        state: eqx.nn.State,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        *,
        key: Key,
        opt_state: optax.OptState,
    ) -> tuple[eqx.nn.State, IterationCarry[ActType, ObsType]]:
        env_state, next_obs, _ = self.env.reset(state.substate(self.env), key=key)
        state = state.update(env_state)

        pol_state = policy.reset(state.substate(policy))
        state = state.update(pol_state)

        return state, IterationCarry(
            StepCarry(jnp.asarray(0), next_obs, jnp.asarray(False), jnp.asarray(False)),
            policy,
            opt_state,
        )

    def iteration(
        self,
        state: eqx.nn.State,
        carry: IterationCarry[ActType, ObsType],
        *,
        key: Key,
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
    ) -> tuple[eqx.nn.State, IterationCarry[ActType, ObsType]]:
        rollout_key, train_key = jr.split(key, 2)
        state, step_carry, rollout_buffer, episode_stats = self.collect_rollout(
            carry.policy, state, carry.step_carry, key=rollout_key
        )
        state, policy, opt_state, log = self.train(
            state, carry.policy, carry.opt_state, rollout_buffer, key=train_key
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

        return state, IterationCarry(step_carry, policy, opt_state)

    def init_tensorboard(
        self,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        tb_log_name: str | bool,
    ) -> JITSummaryWriter | None:
        if tb_log_name is False:
            return None

        if tb_log_name is True:
            tb_log_name = f"logs/{type(policy).__name__}_{self.env.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return JITSummaryWriter(tb_log_name)

    def init_progress_bar(
        self,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        total_timesteps: int,
        show_progress_bar: bool,
    ) -> JITProgressBar | None:
        if show_progress_bar:
            name = f"Training {type(policy).__name__} on {self.env.name}"
            progress_bar = JITProgressBar(name, total=total_timesteps)
            progress_bar.start()
            return progress_bar
        else:
            return None

    def learn(
        self,
        state: eqx.nn.State,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        total_timesteps: int,
        *,
        key: Key,
        show_progress_bar: bool = False,
        tb_log_name: str | bool = False,
    ) -> tuple[eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType]]:
        init_key, learn_key = jr.split(key, 2)

        opt_state = self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array))
        state, carry = self.initialize_iteration_carry(
            state, policy, key=init_key, opt_state=opt_state
        )

        progress_bar = self.init_progress_bar(
            policy, total_timesteps, show_progress_bar
        )
        tb_writer = self.init_tensorboard(policy, tb_log_name)
        num_iterations = total_timesteps // self.num_steps

        def scan_iteration(
            carry: tuple[eqx.nn.State, IterationCarry[ActType, ObsType], Key], _
        ):
            st, it_carry, ky = carry
            iter_key, next_key = jr.split(ky, 2)
            st, it_carry = self.iteration(
                st,
                it_carry,
                key=iter_key,
                progress_bar=progress_bar,
                tb_writer=tb_writer,
            )
            return (st, it_carry, next_key), None

        (state, carry, _), _ = filter_scan(
            scan_iteration, (state, carry, learn_key), length=num_iterations
        )

        if progress_bar is not None:
            progress_bar.stop()

        return state, carry.policy
