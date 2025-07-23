from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Float, Key
from tensorboardX import SummaryWriter

from oryx.buffers import RolloutBuffer
from oryx.env import AbstractEnvLike
from oryx.policies import AbstractActorCriticPolicy

from .base_algorithm import AbstractAlgorithm


class AbstractOnPolicyAlgorithm[ActType, ObsType](AbstractAlgorithm[ActType, ObsType]):
    """Base class for on policy algorithms."""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    policy: eqx.AbstractVar[AbstractActorCriticPolicy[Float, ActType, ObsType]]

    def step(
        self,
        state: eqx.nn.State,
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> tuple[eqx.nn.State, RolloutBuffer[ActType, ObsType]]:
        """Perform a single step in the environment."""
        action_key, env_key, reset_key = jr.split(key, 3)

        raise NotImplementedError

    def collect_rollout(
        self,
        state: eqx.nn.State,
        num_steps: int,
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> tuple[eqx.nn.State, RolloutBuffer[ActType, ObsType]]:
        """Collect a rollout from the environment and store it in a buffer."""
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> AbstractOnPolicyAlgorithm:
        """Train the policy using the rollout buffer."""

    def learn(
        self,
        callback: Callable | None = None,
        *,
        key: Key | None = None,
        progress_bar: bool = False,
        tb_log_name: str | None = None,
        log_interval: int = 100,
    ) -> AbstractOnPolicyAlgorithm[ActType, ObsType]:
        """Return a trained model."""
        raise NotImplementedError
