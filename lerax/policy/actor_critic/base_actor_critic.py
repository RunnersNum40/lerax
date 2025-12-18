from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, Key

from lerax.space import AbstractSpace

from ..base_policy import (
    AbstractPolicy,
    AbstractPolicyState,
    AbstractStatelessPolicy,
    NullPolicyState,
)


class AbstractActorCriticPolicy[StateType: AbstractPolicyState, ActType, ObsType](
    AbstractPolicy[StateType, ActType, ObsType]
):
    """
    Base class for stateful actor-critic policies.

    Actor-critic policies map observations and internal states to actions, values, and new internal states.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def action_and_value(
        self, state: StateType, observation: ObsType, *, key: Key
    ) -> tuple[StateType, ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        Args:
            state: The current policy state.
            observation: The observation to get the action and value for.
            key: A JAX PRNG key.

        Returns:
            new_state: The new policy state.
            action: The action to take.
            value: The value of the observation.
            log_prob: The log probability of the action.
        """

    @abstractmethod
    def evaluate_action(
        self, state: StateType, observation: ObsType, action: ActType
    ) -> tuple[StateType, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """
        Evaluate an action given an observation.

        Args:
            state: The current policy state.
            observation: The observation to evaluate the action for.
            action: The action to evaluate.

        Returns:
            new_state: The new policy state.
            value: The value of the observation.
            log_prob: The log probability of the action.
            entropy: The entropy of the action distribution.
        """

    @abstractmethod
    def value(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Float[Array, ""]]:
        """
        Get the value of an observation.

        Args:
            state: The current policy state.
            observation: The observation to get the value for.

        Returns:
            new_state: The new policy state.
            value: The value of the observation.
        """


class AbstractStatelessActorCriticPolicy[ActType, ObsType](
    AbstractActorCriticPolicy[NullPolicyState, ActType, ObsType],
    AbstractStatelessPolicy[ActType, ObsType],
):
    """
    Base class for stateless actor-critic policies.

    Actor-critic policies map observations to actions and values.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def stateless_action_and_value(
        self, observation: ObsType, *, key: Key
    ) -> tuple[ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        Args:
            observation: The observation to get the action and value for.
            key: A JAX PRNG key.

        Returns:
            action: The action to take.
            value: The value of the observation.
            log_prob: The log probability of the action.
        """

    def action_and_value(
        self, state: NullPolicyState, observation: ObsType, *, key: Key
    ) -> tuple[NullPolicyState, ActType, Float[Array, ""], Float[Array, ""]]:
        return state, *self.stateless_action_and_value(observation, key=key)

    @abstractmethod
    def stateless_evaluate_action(
        self, observation: ObsType, action: ActType
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """
        Evaluate an action given an observation.

        Args:
            observation: The observation to evaluate the action for.
            action: The action to evaluate.

        Returns:
            value: The value of the observation.
            log_prob: The log probability of the action.
            entropy: The entropy of the action distribution.
        """

    def evaluate_action(
        self, state: NullPolicyState, observation: ObsType, action: ActType
    ) -> tuple[NullPolicyState, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        return state, *self.stateless_evaluate_action(observation, action)

    @abstractmethod
    def stateless_value(self, observation: ObsType) -> Float[Array, ""]:
        """
        Get the value of an observation.

        Args:
            observation: The observation to get the value for.

        Returns:
            value: The value of the observation.
        """

    def value(
        self, state: NullPolicyState, observation: ObsType
    ) -> tuple[NullPolicyState, Float[Array, ""]]:
        return state, self.stateless_value(observation)
