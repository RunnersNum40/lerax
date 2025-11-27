from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from lerax.space.base_space import AbstractSpace
from lerax.utils import Serializable


class AbstractPolicyState(eqx.Module):
    pass


class AbstractStatelessPolicy[ActType, ObsType](Serializable):
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
