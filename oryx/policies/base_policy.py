from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from oryx.spaces.base_space import AbstractSpace


# TODO: Break out stateful and non-stateful policies
# TODO: Break out stochastic and non-stochastic policies
class AbstractPolicy[ActType, ObsType](eqx.Module):
    """
    Base class for policies.

    Policies map from observations to actions.
    """

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]

    @property
    @abstractmethod
    def action_space(self) -> AbstractSpace[ActType]:
        """The action space of the policy."""

    @property
    @abstractmethod
    def observation_space(self) -> AbstractSpace[ObsType]:
        """The observation space of the policy."""

    @abstractmethod
    def predict(
        self, state: eqx.nn.State, observation: ObsType, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, ActType]:
        """Choose an action from an observation."""
