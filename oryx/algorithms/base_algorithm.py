from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jaxtyping import Key

from oryx.policies import AbstractPolicy
from oryx.spaces import AbstractSpace


class AbstractAlgorithm(eqx.Module, strict=True):
    """Base class for RL algorithms."""

    policy: eqx.AbstractVar[AbstractPolicy]

    observation_space: eqx.AbstractVar[AbstractSpace]
    action_space: eqx.AbstractVar[AbstractSpace]

    @abstractmethod
    def learn(
        self,
        callback: Callable | None = None,
        *,
        key: Key | None = None,
        progress_bar: bool = False,
        tb_log_name: str,
        log_interval: int = 100,
    ) -> AbstractAlgorithm:
        """Return a trained model."""

    # TODO: Properly type the load method.
    @classmethod
    @abstractmethod
    def load(cls, path, *args, **kwargs) -> AbstractAlgorithm:
        """Load a model from a file."""

    # TODO: Properly type the save method.
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to a file."""
