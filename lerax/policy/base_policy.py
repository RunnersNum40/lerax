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


class AbstractStatelessPolicy[ActType, ObsType](Serializable):
    """
    Base class for stateless policies.

    Policies map observations to actions.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def __call__(self, observation: ObsType, *, key: Key | None = None) -> ActType:
        pass

    @abstractmethod
    def into_stateful[SelfType: AbstractStatelessPolicy](
        self: SelfType,
    ) -> AbstractStatefulWrapper[SelfType, ActType, ObsType]:
        pass


class AbstractStatefulPolicy[StateType: AbstractPolicyState, ActType, ObsType](
    Serializable
):
    """
    Base class for stateful policies.

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
        pass

    @abstractmethod
    def reset(self, *, key: Key) -> StateType:
        pass

    def into_stateful[SelfType: AbstractStatefulPolicy](self: SelfType) -> SelfType:
        return self


class NullStatefulPolicyState(AbstractPolicyState):
    pass


class AbstractStatefulWrapper[PolicyType: AbstractStatelessPolicy, ActType, ObsType](
    AbstractStatefulPolicy[NullStatefulPolicyState, ActType, ObsType]
):
    """
    Wrapper to convert a stateless policy into a stateful policy with a null state.

    Used to enable compatibility between stateless and stateful policy interfaces.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
        policy: The underlying stateless policy.
    """

    policy: eqx.AbstractVar[PolicyType]

    @property
    def name(self) -> str:
        return self.policy.name

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.policy.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.policy.observation_space

    def __call__(
        self,
        state: NullStatefulPolicyState,
        observation: ObsType,
        *,
        key: Key | None = None,
    ) -> tuple[NullStatefulPolicyState, ActType]:
        action = self.policy(observation, key=key)
        return state, action

    def reset(self, *, key: Key) -> NullStatefulPolicyState:
        return NullStatefulPolicyState()

    def into_stateless(self) -> PolicyType:
        return self.policy


type AbstractPolicy = AbstractStatelessPolicy | AbstractStatefulPolicy
