from __future__ import annotations

from abc import abstractmethod

import equinox as eqx

from lerax.space.base_space import AbstractSpace


class AbstractPolicyState(eqx.Module):
    """State container for stateful policies."""


class AbstractPolicy[ActType, ObsType](eqx.Module):
    """
    Base class for policies.

    Policies map from observations to actions.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]


class AbstractStatefulPolicy[StateType: AbstractPolicyState, ActType, ObsType](
    AbstractPolicy[ActType, ObsType]
):
    """
    Base class for stateful policies.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def reset(self, *args, **kwargs) -> StateType:
        """Reset the policy's internal state."""
