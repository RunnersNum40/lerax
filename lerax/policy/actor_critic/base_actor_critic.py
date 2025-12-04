from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, Key

from lerax.space import AbstractSpace

from ..base_policy import (
    AbstractPolicyState,
    AbstractStatefulPolicy,
    AbstractStatefulWrapper,
    AbstractStatelessPolicy,
    NullStatefulPolicyState,
)


class AbstractStatelessActorCriticPolicy[ActType, ObsType](
    AbstractStatelessPolicy[ActType, ObsType]
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
    def action_and_value(
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

    @abstractmethod
    def evaluate_action(
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

    @abstractmethod
    def value(self, observation: ObsType) -> Float[Array, ""]:
        """
        Get the value of an observation.

        Args:
            observation: The observation to get the value for.

        Returns:
            value: The value of the observation.
        """

    def into_stateful[SelfType: AbstractStatelessActorCriticPolicy](
        self: SelfType,
    ) -> ActorCriticStatefulWrapper[SelfType, ActType, ObsType]:
        """
        Transform this stateless policy into a stateful one.

        Returns:
            A stateful wrapper around this policy.
        """
        return ActorCriticStatefulWrapper(self)


class AbstractStatefulActorCriticPolicy[
    StateType: AbstractPolicyState, ActType, ObsType
](AbstractStatefulPolicy[StateType, ActType, ObsType]):
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


class ActorCriticStatefulWrapper[
    PolicyType: AbstractStatelessActorCriticPolicy,
    ActType,
    ObsType,
](
    AbstractStatefulActorCriticPolicy[NullStatefulPolicyState, ActType, ObsType],
    AbstractStatefulWrapper[PolicyType, ActType, ObsType],
):
    policy: PolicyType

    def __init__(self, policy: PolicyType):
        self.policy = policy

    def action_and_value(
        self, state: NullStatefulPolicyState, observation: ObsType, *, key: Key
    ) -> tuple[NullStatefulPolicyState, ActType, Float[Array, ""], Float[Array, ""]]:
        return state, *self.policy.action_and_value(observation, key=key)

    def evaluate_action(
        self, state: NullStatefulPolicyState, observation: ObsType, action: ActType
    ) -> tuple[
        NullStatefulPolicyState, Float[Array, ""], Float[Array, ""], Float[Array, ""]
    ]:
        return state, *self.policy.evaluate_action(observation, action)

    def value(
        self, state: NullStatefulPolicyState, observation: ObsType
    ) -> tuple[NullStatefulPolicyState, Float[Array, ""]]:
        return state, self.policy.value(observation)


type AbstractActorCriticPolicy = AbstractStatefulActorCriticPolicy | AbstractStatelessActorCriticPolicy
