from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

import equinox as eqx
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Int, Key, Scalar

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicy, AbstractPolicyState, AbstractStatefulPolicy
from lerax.utils import filter_scan

from .utils import EpisodeStats, JITProgressBar, JITSummaryWriter


class AbstractStepState(eqx.Module):
    """Base class for algorithm state that is vectorized over environment steps."""

    env_state: eqx.AbstractVar[AbstractEnvLikeState]
    policy_state: eqx.AbstractVar[AbstractPolicyState]
    episode_stats: eqx.AbstractVar[EpisodeStats]


class AbstractAlgorithmState[PolicyType: AbstractStatefulPolicy](eqx.Module):
    """Base class for algorithm states."""

    iteration_count: eqx.AbstractVar[Int[Array, ""]]
    step_state: eqx.AbstractVar[AbstractStepState]
    env: eqx.AbstractVar[AbstractEnvLike]
    policy: eqx.AbstractVar[PolicyType]
    opt_state: eqx.AbstractVar[optax.OptState]
    tb_writer: eqx.AbstractVar[JITSummaryWriter | None]
    progress_bar: eqx.AbstractVar[JITProgressBar | None]

    def next[A: AbstractAlgorithmState](
        self: A,
        step_state: AbstractStepState,
        policy: PolicyType,
        opt_state: optax.OptState,
    ) -> A:
        return eqx.tree_at(
            lambda s: (s.iteration_count, s.step_state, s.policy, s.opt_state),
            self,
            (self.iteration_count + 1, step_state, policy, opt_state),
        )


class AbstractAlgorithm[
    PolicyType: AbstractStatefulPolicy, StateType: AbstractAlgorithmState
](eqx.Module):
    """Base class for RL algorithms."""

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]

    @abstractmethod
    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key,
        tb_writer: JITSummaryWriter | None,
        progress_bar: JITProgressBar | None,
    ) -> StateType:
        """Return the initial carry for the training iteration."""

    @abstractmethod
    def per_iteration(self, state: StateType) -> StateType:
        """Process the algorithm state after each iteration."""

    @staticmethod
    def init_tensorboard(
        env: AbstractEnvLike,
        policy: AbstractPolicy,
        tb_log: str | bool,
    ) -> JITSummaryWriter | None:
        if tb_log is False:
            return None

        if tb_log is True:
            tb_log = f"logs/{policy.name}_{env.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return JITSummaryWriter(tb_log)

    def write_tensorboard(
        self,
        state: StateType,
        log: dict[str, Scalar],
        episode_stats: EpisodeStats,
    ) -> None:
        tb_writer = state.tb_writer
        last_step = state.iteration_count * self.num_envs * self.num_steps - 1
        first_step = last_step - self.num_envs * self.num_steps + 1
        if tb_writer is not None:
            log["learning_rate"] = optax.tree_utils.tree_get(
                state.opt_state, "learning_rate", jnp.nan
            )
            tb_writer.add_dict(log, prefix="train", global_step=last_step)
            tb_writer.log_episode_stats(episode_stats, first_step=first_step)

    @staticmethod
    def init_progress_bar(
        env: AbstractEnvLike,
        policy: AbstractPolicy,
        total_timesteps: int,
        show_progress_bar: bool,
    ) -> JITProgressBar | None:
        if show_progress_bar:
            name = f"Training {policy.name} on {env.name}"
            progress_bar = JITProgressBar(name, total=total_timesteps)
            progress_bar.start()
            return progress_bar
        else:
            return None

    def update_progress_bar(self, state: StateType) -> None:
        progress_bar = state.progress_bar
        if progress_bar is not None:
            progress_bar.update(advance=self.num_envs * self.num_steps)

    @abstractmethod
    def iteration(
        self,
        state: StateType,
        *,
        key: Key,
    ) -> StateType:
        """
        Perform a single iteration of training.

        **Arguments:**
            - state: The current algorithm state.
            - key: A JAX Key.

        **Returns:**
            - state: The updated algorithm state.
        """

    @abstractmethod
    def num_iterations(self, total_timesteps: int) -> int:
        """Number of iterations per training session."""

    @eqx.filter_jit
    def _learn(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        total_timesteps: int,
        *,
        key: Key,
        tb_writer: JITSummaryWriter | None = None,
        progress_bar: JITProgressBar | None = None,
    ) -> PolicyType:
        """
        Wrapper around the learning process.

        The scan will already lower the code but using eqx.filter_jit provides
        better error messages.
        """
        reset_key, learn_key = jr.split(key, 2)

        state = self.reset(
            env, policy, key=reset_key, tb_writer=tb_writer, progress_bar=progress_bar
        )

        state, _ = filter_scan(
            lambda s, k: (self.iteration(s, key=k), None),
            state,
            jr.split(learn_key, self.num_iterations(total_timesteps)),
        )

        return state.policy

    # TODO: Add support for callbacks
    def learn[A: AbstractPolicy](
        self,
        env: AbstractEnvLike,
        policy: A,
        total_timesteps: int,
        *,
        key: Key,
        show_progress_bar: bool = False,
        tb_log: str | bool = False,
    ) -> A:
        """
        Train the policy on the environment for a given number of timesteps.

        **Arguments:**
            - env: The environment to train on.
            - policy: The policy to train.
            - total_timesteps: The total number of timesteps to train for.
            - key: A JAX Key.
            - show_progress_bar: Whether to show a progress bar during training.
            - tb_log: The TensorBoard log directory or True, or False to disable logging.

        **Returns:**
            - policy: The trained policy.
        """
        progress_bar = self.init_progress_bar(
            env, policy, total_timesteps, show_progress_bar
        )
        tb_writer = self.init_tensorboard(env, policy, tb_log)

        # TODO: Revisit the typing here
        if isinstance(policy, AbstractStatefulPolicy):
            return self._learn(
                env,
                policy,  # pyright: ignore
                total_timesteps,
                key=key,
                progress_bar=progress_bar,
                tb_writer=tb_writer,
            )
        else:
            return self._learn(
                env,
                policy.into_stateful(),  # pyright: ignore
                total_timesteps,
                key=key,
                progress_bar=progress_bar,
                tb_writer=tb_writer,
            ).into_stateless()  # pyright: ignore
