from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

import equinox as eqx
import optax
from jax import random as jr
from jaxtyping import Array, Int, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicy, AbstractPolicyState, AbstractStatefulPolicy
from lerax.utils import filter_scan

from .utils import EpisodeStats, JITProgressBar, JITSummaryWriter


class AbstractStepCarry(eqx.Module):
    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    episode_stats: EpisodeStats


class AbstractIterationCarry[PolicyType: AbstractStatefulPolicy](eqx.Module):
    iteration_count: Int[Array, ""]
    step_carry: AbstractStepCarry
    policy: PolicyType
    opt_state: optax.OptState


class AbstractAlgorithm(eqx.Module):
    """Base class for RL algorithms."""

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]

    @abstractmethod
    def init_iteration_carry[A: AbstractStatefulPolicy](
        self,
        env: AbstractEnvLike,
        policy: A,
        *,
        key: Key,
    ) -> AbstractIterationCarry[A]:
        """Return the initial carry for the training iteration."""

    def init_tensorboard(
        self,
        env: AbstractEnvLike,
        policy: AbstractPolicy,
        tb_log: str | bool,
    ) -> JITSummaryWriter | None:
        if tb_log is False:
            return None

        if tb_log is True:
            tb_log = f"logs/{policy.name}_{env.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return JITSummaryWriter(tb_log)

    def init_progress_bar(
        self,
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

    @abstractmethod
    def iteration(
        self,
        env: AbstractEnvLike,
        carry,  # Generic typing, eugh :(
        *,
        key: Key,
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
    ) -> AbstractIterationCarry:
        """Perform a single iteration of training."""

    @eqx.filter_jit
    def _learn[A: AbstractStatefulPolicy](
        self,
        env: AbstractEnvLike,
        policy: A,
        total_timesteps: int,
        *,
        key: Key,
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
    ) -> A:
        init_key, learn_key = jr.split(key, 2)
        carry = self.init_iteration_carry(env, policy, key=init_key)
        num_iterations = total_timesteps // (self.num_steps * self.num_envs)

        def scan_iteration(carry: tuple[AbstractIterationCarry, Key], _):
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

        return carry.policy

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

        if isinstance(policy, AbstractStatefulPolicy):
            return self._learn(
                env,
                policy,
                total_timesteps,
                key=key,
                progress_bar=progress_bar,
                tb_writer=tb_writer,
            )
        else:
            return self._learn(
                env,
                policy.into_stateful(),
                total_timesteps,
                key=key,
                progress_bar=progress_bar,
                tb_writer=tb_writer,
            ).into_stateless()
