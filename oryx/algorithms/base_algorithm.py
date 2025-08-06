from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from oryx.env import AbstractEnvLike
from oryx.policies import AbstractPolicy


class AbstractAlgorithm[ActType, ObsType](eqx.Module):
    """Base class for RL algorithms."""

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]
    policy: eqx.AbstractVar[AbstractPolicy]
    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    # TODO: Add support for callbacks
    @abstractmethod
    def learn(
        self,
        state: eqx.nn.State,
        total_timesteps: int,
        *,
        key: Key,
        progress_bar: bool = False,
        tb_log_name: str | None = None,
        log_interval: int = 100,
    ):
        """Return a trained model."""

    @classmethod
    @abstractmethod
    def load(cls, path, *args, **kwargs) -> AbstractAlgorithm[ActType, ObsType]:
        """Load a model from a file."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to a file."""
