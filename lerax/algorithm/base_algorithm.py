from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import optax
from jaxtyping import Key

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy


class AbstractAlgorithm[ActType, ObsType](eqx.Module):
    """Base class for RL algorithms."""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    policy: eqx.AbstractVar[AbstractPolicy]
    optimizer: eqx.AbstractVar[optax.GradientTransformation]
    opt_state_index: eqx.AbstractVar[eqx.nn.StateIndex[optax.OptState]]

    # TODO: Add support for callbacks
    @abstractmethod
    def learn(
        self,
        state: eqx.nn.State,
        total_timesteps: int,
        *,
        key: Key,
        show_progress_bar: bool = False,
        tb_log_name: str | None = None,
    ) -> tuple[eqx.nn.State, AbstractPolicy[ActType, ObsType]]:
        """Return a trained model."""

    @classmethod
    @abstractmethod
    def load(cls, path, *args, **kwargs) -> AbstractAlgorithm[ActType, ObsType]:
        """Load a model from a file."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to a file."""
