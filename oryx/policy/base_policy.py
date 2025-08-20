from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from oryx.space.base_space import AbstractSpace


# TODO: Break out stateful and non-stateful policies
# TODO: Break out stochastic and non-stochastic policies
class AbstractPolicy[ActType, ObsType](eqx.Module):
    """
    Base class for policies.

    Policies map from observations to actions.
    """

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def predict(
        self, state: eqx.nn.State, observation: ObsType, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, ActType]:
        """Choose an action from an observation."""
