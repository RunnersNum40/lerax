from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Key

from lerax.env import AbstractEnvLike
from lerax.policy import (
    AbstractPolicy,
    AbstractStatefulPolicy,
    AbstractStatelessPolicy,
)
from lerax.utils import filter_scan

from .utils import IterationCarry, JITProgressBar, JITSummaryWriter, StepCarry


class AbstractAlgorithm[PolicyType: AbstractPolicy](eqx.Module):
    """Base class for RL algorithms."""

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]

    def init_iteration_carry[WrappedPolicyType: AbstractStatefulPolicy](
        self,
        env: AbstractEnvLike,
        policy: WrappedPolicyType,
        *,
        key: Key,
    ) -> IterationCarry[WrappedPolicyType]:
        if self.num_envs == 1:
            step_carry = StepCarry.initial(env, policy, key)
        else:
            step_carry = jax.vmap(StepCarry.initial, in_axes=(None, None, 0))(
                env, policy, jr.split(key, self.num_envs)
            )

        return IterationCarry(
            jnp.array(0, dtype=int),
            step_carry,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
        )

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
    def iteration[WrappedPolicyType: AbstractStatefulPolicy](
        self,
        env: AbstractEnvLike,
        carry: IterationCarry[WrappedPolicyType],
        *,
        key: Key,
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
    ) -> IterationCarry[WrappedPolicyType]:
        """Perform a single iteration of training."""

    # TODO: Add support for callbacks
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
        if not isinstance(policy, AbstractStatefulPolicy):
            wrapped_policy = policy.into_stateful()
        else:
            wrapped_policy = policy

        init_key, learn_key = jr.split(key, 2)

        carry = self.init_iteration_carry(env, wrapped_policy, key=init_key)

        progress_bar = self.init_progress_bar(
            env, wrapped_policy, total_timesteps, show_progress_bar
        )
        tb_writer = self.init_tensorboard(env, wrapped_policy, tb_log)
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

        if isinstance(policy, AbstractStatelessPolicy):
            return carry.policy.policy
        else:
            return carry.policy
