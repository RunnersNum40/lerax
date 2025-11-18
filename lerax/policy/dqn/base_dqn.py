from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Integer, Key

from lerax.space import AbstractSpace, Discrete

from ..base_policy import (
    AbstractPolicyState,
    AbstractStatefulPolicy,
    AbstractStatefulWrapper,
    AbstractStatelessPolicy,
    NullStatefulPolicyState,
)
from ..q import AbstractStatefulQPolicy, AbstractStatelessQPolicy, QStatefulWrapper


class AbstractStatelessDQNPolicy[ObsType, PolicyType: AbstractStatelessQPolicy](
    AbstractStatelessPolicy[Integer[Array, ""], ObsType]
):
    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    q_network: PolicyType
    target_q_network: PolicyType

    def q_values(self, observation: ObsType) -> Array:
        return self.q_network.q_values(observation)

    def target_q_values(self, observation: ObsType) -> Array:
        return self.target_q_network.q_values(observation)

    def __call__(
        self, observation: ObsType, *, key: Key | None = None
    ) -> Integer[Array, ""]:
        return self.q_network(observation, key=key)

    def into_stateful[SelfType: AbstractStatelessDQNPolicy](
        self: SelfType,
    ) -> DQNStatefulWrapper[SelfType, ObsType]:
        return DQNStatefulWrapper(self)


class AbstractStatefulDQNPolicy[
    StateType: AbstractPolicyState, ObsType, PolicyType: AbstractStatefulQPolicy
](AbstractStatefulPolicy[StateType, Integer[Array, ""], ObsType]):
    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    q_network: PolicyType
    target_q_network: PolicyType

    def __call__(
        self, state: StateType, observation: ObsType, *, key: Key | None = None
    ) -> tuple[StateType, Integer[Array, ""]]:
        return self.q_network(state, observation, key=key)

    def q_values(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Array]:
        return self.q_network.q_values(state, observation)

    def target_q_values(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Array]:
        return self.target_q_network.q_values(state, observation)


class DQNStatefulWrapper[PolicyType: AbstractStatelessDQNPolicy, ObsType](
    AbstractStatefulDQNPolicy[
        NullStatefulPolicyState, ObsType, AbstractStatefulQPolicy
    ],
    AbstractStatefulWrapper[PolicyType, Integer[Array, ""], ObsType],
):
    _policy: PolicyType
    q_network: AbstractStatefulQPolicy
    target_q_network: AbstractStatefulQPolicy

    def __init__(self, policy: PolicyType):
        self._policy = policy
        self.q_network = QStatefulWrapper(policy.q_network)
        self.target_q_network = QStatefulWrapper(policy.target_q_network)

    def reset(self, *, key):
        return NullStatefulPolicyState()

    @property
    def policy(self) -> PolicyType:
        return eqx.tree_at(
            lambda p: (p.q_network, p.target_q_network),
            self._policy,
            (self.q_network, self.target_q_network),
        )
