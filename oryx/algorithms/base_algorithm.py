from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jaxtyping import Key

from oryx.env import AbstractEnvLike
from oryx.policies import AbstractPolicy


class AbstractAlgorithm[ActType, ObsType](eqx.Module):
    """Base class for RL algorithms."""

    policy: eqx.AbstractVar[AbstractPolicy]
    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    @abstractmethod
    def learn(
        self,
        callback: Callable | None = None,
        *,
        key: Key | None = None,
        progress_bar: bool = False,
        tb_log_name: str | None = None,
        log_interval: int = 100,
    ) -> AbstractAlgorithm[ActType, ObsType]:
        """Return a trained model."""

    # TODO: Properly type the load method.
    @classmethod
    @abstractmethod
    def load(cls, path, *args, **kwargs) -> AbstractAlgorithm[ActType, ObsType]:
        """Load a model from a file."""

    # TODO: Properly type the save method.
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to a file."""
