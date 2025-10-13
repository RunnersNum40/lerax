from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy


class AbstractAlgorithm[ActType, ObsType](eqx.Module):
    """Base class for RL algorithms."""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    # TODO: Add support for callbacks
    @abstractmethod
    def learn(
        self,
        policy: AbstractPolicy[ActType, ObsType],
        *args,
        key: Key,
        **kwargs,
    ) -> tuple[eqx.nn.State, AbstractPolicy[ActType, ObsType]]:
        """Train and return an updated policy."""

    @classmethod
    @abstractmethod
    def load(cls, path, *args, **kwargs) -> AbstractAlgorithm[ActType, ObsType]:
        """Load a model from a file."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to a file."""
