from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicy, AbstractPolicyState


class AbstractAlgorithm[
    EnvStateType: AbstractEnvLikeState,
    PolicyStateType: AbstractPolicyState,
    ActType,
    ObsType,
](eqx.Module):
    """Base class for RL algorithms."""

    # TODO: Add support for callbacks
    @abstractmethod
    def learn(
        self,
        env: AbstractEnvLike[EnvStateType, ActType, ObsType],
        policy: AbstractPolicy[PolicyStateType, ActType, ObsType],
        *args,
        key: Key,
        **kwargs,
    ) -> AbstractPolicy[PolicyStateType, ActType, ObsType]:
        """Train and return an updated policy."""
