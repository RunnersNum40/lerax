from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, Key

from lerax.space import AbstractSpace

from ..base_policy import (
    AbstractPolicy,
    AbstractPolicyState,
    AbstractStatefulPolicy,
    AbstractStatefulWrapper,
    AbstractStatelessPolicy,
    NullStatefulPolicyState,
)


class AbstractActorCriticPolicy[ActType, ObsType](AbstractPolicy[ActType, ObsType]):
    """Base class for actor-critic policies."""

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]


class AbstractStatelessActorCriticPolicy[
    ActType,
    ObsType,
](
    AbstractActorCriticPolicy[ActType, ObsType],
    AbstractStatelessPolicy[ActType, ObsType],
):
    """
    Base class for stateless actor-critic policies.

    This class is intended for policies that do not maintain an internal state.
    Stateless policies can be converted to stateful ones using into_stateful().
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def __call__(self, observation: ObsType, *, key: Key | None = None) -> ActType:
        """
        Get an action from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """

    @abstractmethod
    def action_and_value(
        self, observation: ObsType, *, key: Key
    ) -> tuple[ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """

    @abstractmethod
    def evaluate_action(
        self, observation: ObsType, action: ActType
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Evaluate an action given an observation."""

    @abstractmethod
    def value(self, observation: ObsType) -> Float[Array, ""]:
        """Get the value of an observation."""

    def into_stateful[SelfType: AbstractStatelessActorCriticPolicy](
        self: SelfType,
    ) -> ActorCriticStatefulWrapper[SelfType, ActType, ObsType]:
        return ActorCriticStatefulWrapper(self)


class AbstractStatefulActorCriticPolicy[
    StateType: AbstractPolicyState, ActType, ObsType
](
    AbstractActorCriticPolicy[ActType, ObsType],
    AbstractStatefulPolicy[StateType, ActType, ObsType],
):
    """
    Base class for stateful actor-critic policies.

    This class is intended for policies that maintain an internal state, such as RNNs.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def __call__(
        self, state: StateType, observation: ObsType, *, key: Key | None = None
    ) -> tuple[StateType, ActType]:
        """
        Get an action from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """

    @abstractmethod
    def action_and_value(
        self, state: StateType, observation: ObsType, *, key: Key
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

    def __call__(
        self,
        state: NullStatefulPolicyState,
        observation: ObsType,
        *,
        key: Key | None = None,
    ) -> tuple[NullStatefulPolicyState, ActType]:
        return state, self.policy(observation, key=key)

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
