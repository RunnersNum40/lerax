from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from lerax.space.base_space import AbstractSpace


# TODO: Break out stateful and non-stateful policies
# TODO: Break out stochastic and non-stochastic policies
class AbstractPolicy[ActType, ObsType](eqx.Module):
    """
    Base class for policies.

    Policies map from observations to actions.
    """

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def predict(
        self,
        state: eqx.nn.State,
        observation: ObsType,
        *args,
        key: Key | None = None,
        **kwargs,
    ) -> tuple[eqx.nn.State, ActType]:
        """Choose an action from an observation."""

    @abstractmethod
    def reset(self, state: eqx.nn.State, *args, **kwargs) -> eqx.nn.State:
        """Reset the policy's internal state."""
