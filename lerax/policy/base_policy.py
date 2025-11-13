from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key

from lerax.space.base_space import AbstractSpace


class AbstractPolicyState(eqx.Module):
    pass


class AbstractPolicy[ActType, ObsType](eqx.Module):
    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def into_stateful(self) -> AbstractStatefulPolicy:
        pass


class AbstractStatelessPolicy[ActType, ObsType](AbstractPolicy[ActType, ObsType]):
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
    AbstractPolicy[ActType, ObsType]
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
    def reset(self) -> StateType:
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
        return NullStatefulPolicyState(), action

    def reset(self) -> NullStatefulPolicyState:
        return NullStatefulPolicyState()
