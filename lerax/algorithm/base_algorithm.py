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
from lerax.policy import AbstractPolicy, AbstractStatefulPolicy

from .utils import IterationCarry, JITProgressBar, JITSummaryWriter, StepCarry


class AbstractAlgorithm(eqx.Module):
    """Base class for RL algorithms."""

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    num_envs: eqx.AbstractVar[int]

    def init_iteration_carry(
        self,
        env: AbstractEnvLike,
        policy: AbstractStatefulPolicy,
        *,
        key: Key,
    ) -> IterationCarry:
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

    # TODO: Add support for callbacks
    @abstractmethod
    def learn[PolicyType: AbstractPolicy](
        self, env: AbstractEnvLike, policy: PolicyType, *args, key: Key, **kwargs
    ) -> PolicyType:
        """Train and return an updated policy."""
