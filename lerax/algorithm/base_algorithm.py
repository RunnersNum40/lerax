from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy


class AbstractAlgorithm(eqx.Module):
    """Base class for RL algorithms."""

    # TODO: Add support for callbacks
    @abstractmethod
    def learn[PolicyType: AbstractPolicy](
        self, env: AbstractEnvLike, policy: PolicyType, *args, key: Key, **kwargs
    ) -> PolicyType:
        """Train and return an updated policy."""
