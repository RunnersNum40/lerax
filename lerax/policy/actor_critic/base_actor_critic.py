from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, Key

from lerax.space import AbstractSpace

from ..base_policy import AbstractPolicy, AbstractPolicyState


class AbstractActorCriticPolicy[StateType: AbstractPolicyState, ActType, ObsType](
    AbstractPolicy[StateType, ActType, ObsType]
):
    """
    Base class for actor-critic policies.

    For now all policies are treated as stateful, meaning they maintain an internal
    state. Non-stateful policies can simply ignore the state during implementation.
    """

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def action_and_value(
        self, state: StateType, observation: ObsType, *, key: Key | None = None
    ) -> tuple[StateType, ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """

    @abstractmethod
    def evaluate_action(
        self, state: StateType, observation: ObsType, action: ActType
    ) -> tuple[StateType, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Evaluate an action given an observation."""

    @abstractmethod
    def value(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Float[Array, ""]]:
        """Get the value of an observation."""
