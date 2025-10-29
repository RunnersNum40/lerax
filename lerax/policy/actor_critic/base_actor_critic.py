from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, Key

from lerax.space import AbstractSpace

from ..base_policy import AbstractPolicy, AbstractPolicyState, AbstractStatefulPolicy


class NullStatefulActorCriticPolicyState(AbstractPolicyState):
    """Marker class for stateless actor-critic policies."""


class AbstractActorCriticPolicy[ActType, ObsType](AbstractPolicy[ActType, ObsType]):
    """Base class for actor-critic policies."""


class AbstractStatelessActorCriticPolicy[
    ActType,
    ObsType,
](
    AbstractActorCriticPolicy[ActType, ObsType],
):
    """
    Base class for stateless actor-critic policies.

    This class is intended for policies that do not maintain an internal state.
    Internally stateless policies can be converted to stateful ones using into_stateful().
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def action_and_value(
        self, observation: ObsType, *, key: Key | None = None
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

    def into_stateful(
        self,
    ) -> StatefulWrapper[ActType, ObsType]:
        """Convert this stateless policy into a stateful one."""
        return StatefulWrapper(self)


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


class StatefulWrapper[ActType, ObsType](
    AbstractStatefulActorCriticPolicy[
        NullStatefulActorCriticPolicyState, ActType, ObsType
    ]
):
    policy: AbstractStatelessActorCriticPolicy[ActType, ObsType]

    def __init__(self, policy: AbstractStatelessActorCriticPolicy[ActType, ObsType]):
        self.policy = policy

    @property
    def name(self) -> str:
        return self.policy.name

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.policy.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.policy.observation_space

    def action_and_value(
        self,
        state: NullStatefulActorCriticPolicyState,
        observation: ObsType,
        *,
        key: Key | None = None,
    ) -> tuple[
        NullStatefulActorCriticPolicyState, ActType, Float[Array, ""], Float[Array, ""]
    ]:
        """
        Get an action and value from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """
        return state, *self.policy.action_and_value(observation, key=key)

    def evaluate_action(
        self,
        state: NullStatefulActorCriticPolicyState,
        observation: ObsType,
        action: ActType,
    ) -> tuple[
        NullStatefulActorCriticPolicyState,
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        """Evaluate an action given an observation."""
        return state, *self.policy.evaluate_action(observation, action)

    def value(
        self, state: NullStatefulActorCriticPolicyState, observation: ObsType
    ) -> tuple[NullStatefulActorCriticPolicyState, Float[Array, ""]]:
        """Get the value of an observation."""
        return state, self.policy.value(observation)

    def reset(self) -> NullStatefulActorCriticPolicyState:
        """Reset the policy state."""
        return NullStatefulActorCriticPolicyState()

    def into_stateless(self) -> AbstractStatelessActorCriticPolicy[ActType, ObsType]:
        """Convert this stateful policy into a stateless one."""
        return self.policy
