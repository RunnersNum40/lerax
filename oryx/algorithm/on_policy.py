from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Int, Key, Scalar

from oryx.buffer import RolloutBuffer
from oryx.env import AbstractEnvLike
from oryx.policy import AbstractActorCriticPolicy
from oryx.utils import (
    clone_state,
    filter_scan,
)

from .base_algorithm import AbstractAlgorithm
from .utils import EpisodeStatisticsAccumulator, JITProgressBar, JITSummaryWriter


class StepCarry[ObsType](eqx.Module):
    """Carry for the step function."""

    step_count: Int[Array, ""]
    last_obs: ObsType
    last_termination: Bool[Array, ""]
    last_truncation: Bool[Array, ""]


class IterationCarry[ActType, ObsType](eqx.Module):
    """Carry for the epoch function."""

    step_carry: StepCarry[ObsType]
    policy: AbstractActorCriticPolicy[Float, ActType, ObsType]


class AbstractOnPolicyAlgorithm[ActType, ObsType](AbstractAlgorithm[ActType, ObsType]):
    """
    Base class for on policy algorithms.
    """

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    policy: eqx.AbstractVar[AbstractActorCriticPolicy[Float, ActType, ObsType]]
    optimizer: eqx.AbstractVar[optax.GradientTransformation]
    opt_state_index: eqx.AbstractVar[eqx.nn.StateIndex[optax.OptState]]

    gae_lambda: eqx.AbstractVar[float]
    gamma: eqx.AbstractVar[float]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def __check_init__(self):
        """
        Check invariants.

        Called automatically by Equinox after the object is initialized.
        """
        if (self.env.action_space != self.policy.action_space) or (
            self.env.observation_space != self.policy.observation_space
        ):
            raise ValueError(
                "The action and observation spaces of the environment and policy must match."
            )

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
        """
        Perform a single step in the environment.
        """
        action_key, env_key, reset_key = jr.split(key, 3)

        policy_state = state.substate(self.policy)
        policy_state, action, value, log_prob = policy(
            policy_state,
            carry.last_obs,
            key=action_key,
        )
        state = state.update(policy_state)

        env_state = state.substate(self.env)
        env_state, observation, reward, termination, truncation, info = self.env.step(
            env_state, action, key=env_key
        )
        state = state.update(env_state)

        if "episode" in info:
            episode_stats = EpisodeStatisticsAccumulator.from_episode_stats(
                info["episode"]
            )
        else:
            episode_stats = None

        def reset_env(
            state: eqx.nn.State,
        ) -> tuple[eqx.nn.State, ObsType, dict]:
            """
            Reset the environment and policy states.

            This is called when the episode is done.
            """
            env_state = state.substate(self.env)
            env_state, reset_observation, reset_info = self.env.reset(
                env_state, key=reset_key
            )
            state = state.update(env_state)

            policy_state = state.substate(self.policy)
            policy_state = policy.reset(policy_state)
            state = state.update(policy_state)

            return (
                state,
                reset_observation,
                reset_info,
            )

        def identity(
            state: eqx.nn.State,
        ) -> tuple[eqx.nn.State, ObsType, dict]:
            """
            Return the current state.

            Matches the signature of `reset_env` for use in `lax.cond`.
            """
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
                states=state.substate(self.policy),
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
        """
        Collect a rollout from the environment and store it in a buffer.
        """

        def scan_step(
            carry: tuple[eqx.nn.State, StepCarry[ObsType], Key],
            _,
        ) -> tuple[
            tuple[eqx.nn.State, StepCarry[ObsType], Key],
            tuple[RolloutBuffer[ActType, ObsType], EpisodeStatisticsAccumulator | None],
        ]:
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
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
    ) -> tuple[
        eqx.nn.State,
        AbstractActorCriticPolicy[Float, ActType, ObsType],
        dict[str, Scalar],
    ]:
        """
        Train the policy using the rollout buffer.
        """

    def initialize_iteration_carry(
        self,
        state: eqx.nn.State,
        *,
        key: Key,
    ) -> tuple[eqx.nn.State, IterationCarry[ActType, ObsType]]:
        env_state, next_obs, _ = self.env.reset(state.substate(self.env), key=key)
        state = state.update(env_state)

        policy_state = self.policy.reset(state.substate(self.policy))
        state = state.update(policy_state)

        return state, IterationCarry(
            StepCarry(jnp.asarray(0), next_obs, jnp.asarray(False), jnp.asarray(False)),
            self.policy,
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
        """
        Perform a single iteration of the algorithm.
        """
        rollout_key, train_key = jr.split(key, 2)
        state, step_carry, rollout_buffer, episode_stats = self.collect_rollout(
            carry.policy,
            state,
            carry.step_carry,
            key=rollout_key,
        )
        state, policy, log = self.train(
            state, carry.policy, rollout_buffer, key=train_key
        )

        if progress_bar is not None:
            progress_bar.update(advance=self.num_steps)
        if tb_writer is not None:
            log["learning_rate"] = optax.tree_utils.tree_get(
                state.get(self.opt_state_index), "learning_rate"
            )

            tb_writer.add_dict(
                log, prefix="train", global_step=carry.step_carry.step_count
            )
            if episode_stats is not None:
                tb_writer.log_episode_stats(
                    episode_stats,
                    global_step=carry.step_carry.step_count,
                )

        return state, IterationCarry(step_carry, policy)

    def learn(
        self,
        state: eqx.nn.State,
        total_timesteps: int,
        *,
        key: Key,
        show_progress_bar: bool = False,
        tb_log_name: str | None = None,
    ) -> tuple[eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType]]:
        """
        Return a trained model.
        """

        def scan_iteration(
            carry: tuple[eqx.nn.State, IterationCarry[ActType, ObsType], Key], _
        ) -> tuple[tuple[eqx.nn.State, IterationCarry[ActType, ObsType], Key], None]:
            state, iter_carry, key = carry
            iter_key, carry_key = jr.split(key, 2)

            state, iter_carry = self.iteration(
                state,
                iter_carry,
                key=iter_key,
                progress_bar=progress_bar,
                tb_writer=tb_writer,
            )

            return (state, iter_carry, carry_key), None

        init_key, learn_key = jr.split(key, 2)
        state, carry = self.initialize_iteration_carry(state, key=init_key)

        progress_bar_name = f"Training {type(self.policy).__name__} on {self.env.name}"
        progress_bar = (
            JITProgressBar(progress_bar_name, total=total_timesteps)
            if show_progress_bar
            else None
        )
        tb_writer = JITSummaryWriter(tb_log_name) if tb_log_name is not None else None

        num_iterations = total_timesteps // self.num_steps

        if progress_bar is not None:
            progress_bar.start()

        (state, carry, _), _ = filter_scan(
            scan_iteration, (state, carry, learn_key), length=num_iterations
        )

        if progress_bar is not None:
            progress_bar.stop()

        return state, carry.policy
