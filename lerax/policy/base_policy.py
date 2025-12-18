from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from lerax.space.base_space import AbstractSpace
from lerax.utils import Serializable


class AbstractPolicyState(eqx.Module):
    """
    Base class for policy internal states.
    """

    pass


class AbstractPolicy[StateType: AbstractPolicyState, ActType, ObsType](Serializable):
    """
    Base class for policies.

    Policies map observations and internal states to actions and new internal states.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def __call__(
        self, state: StateType, observation: ObsType, *, key: Key | None = None
    ) -> tuple[StateType, ActType]:
        """
        Return the next action and new internal state given the current
        observation and internal state.

        A key can be provided for stochastic policies. If no key is provided,
        the policy should behave deterministically.

        Args:
            state: The current internal state of the policy.
            observation: The current observation.
            key: An optional JAX random key for stochastic policies.

        Returns:
            The new internal state and the action to take.
        """
        pass

    @abstractmethod
    def reset(self, *, key: Key) -> StateType:
        """
        Return an initial internal state for the policy.

        Args:
            key: A JAX random key for initializing the state.

        Returns:
            An initial internal state for the policy.
        """
        pass


class NullPolicyState(AbstractPolicyState):
    pass


class AbstractStatelessPolicy[ActType, ObsType](
    AbstractPolicy[NullPolicyState, ActType, ObsType]
):
    """
    Base class for stateless policies.

    Wraps a stateless policy into the stateful policy interface by using a
    placeholder.

    To implement a stateless policy, implement the `stateless_call` method
    instead of the `__call__` method.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def stateless_call(
        self, observation: ObsType, *, key: Key | None = None
    ) -> ActType:
        pass

    def __call__(
        self, state: NullPolicyState, observation: ObsType, *, key: Key | None = None
    ) -> tuple[NullPolicyState, ActType]:
        action = self.stateless_call(observation, key=key)
        return state, action

    def reset(self, *, key: Key) -> NullPolicyState:
        return NullPolicyState()
