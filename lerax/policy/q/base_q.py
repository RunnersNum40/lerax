from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import lax
from jax import random as jr
from jaxtyping import Array, Float, Integer

from lerax.space import AbstractSpace, Discrete

from ..base_policy import (
    AbstractPolicyState,
    AbstractStatefulPolicy,
    AbstractStatefulWrapper,
    AbstractStatelessPolicy,
    NullStatefulPolicyState,
)


class AbstractStatelessQPolicy[ObsType](
    AbstractStatelessPolicy[Integer[Array, ""], ObsType]
):
    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    epsilon: eqx.AbstractVar[float]

    @abstractmethod
    def q_values(self, observation: ObsType) -> Float[Array, " actions"]:
        """Return Q-values for all actions given an observation."""

    def into_stateful[SelfType: AbstractStatelessQPolicy](
        self: SelfType,
    ) -> QStatefulWrapper[SelfType, ObsType]:
        return QStatefulWrapper(self)

    def __call__(
        self, observation: ObsType, *, key: Array | None = None
    ) -> Integer[Array, ""]:
        q_vals = self.q_values(observation)

        if key is None or self.epsilon <= 0.0:
            return q_vals.argmax(axis=-1)
        else:
            epsilon_key, action_key = jr.split(key, 2)
            return lax.cond(
                jr.uniform(epsilon_key, shape=()) < self.epsilon,
                lambda: jr.randint(
                    action_key, shape=(), minval=0, maxval=self.action_space.n
                ),
                lambda: q_vals.argmax(axis=-1),
            )


class AbstractStatefulQPolicy[StateType: AbstractPolicyState, ObsType](
    AbstractStatefulPolicy[StateType, Integer[Array, ""], ObsType]
):
    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    epsilon: eqx.AbstractVar[float]

    @abstractmethod
    def q_values(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Float[Array, " actions"]]:
        """Return Q-values for all actions given an observation and state."""

    def __call__(
        self, state: StateType, observation: ObsType, *, key: Array | None = None
    ) -> tuple[StateType, Integer[Array, ""]]:
        state, q_vals = self.q_values(state, observation)

        if key is None or self.epsilon <= 0.0:
            return state, q_vals.argmax(axis=-1)
        else:
            epsilon_key, action_key = jr.split(key, 2)
            action = lax.cond(
                jr.uniform(epsilon_key, shape=()) < self.epsilon,
                lambda: jr.randint(
                    action_key, shape=(), minval=0, maxval=self.action_space.n
                ),
                lambda: q_vals.argmax(axis=-1),
            )
            return state, action


class QStatefulWrapper[PolicyType: AbstractStatelessQPolicy, ObsType](
    AbstractStatefulQPolicy[NullStatefulPolicyState, ObsType],
    AbstractStatefulWrapper[PolicyType, Integer[Array, ""], ObsType],
):
    policy: PolicyType

    def __init__(self, policy: PolicyType):
        self.policy = policy

    @property
    def epsilon(self) -> float:
        return self.policy.epsilon

    def q_values(
        self, state: NullStatefulPolicyState, observation: ObsType
    ) -> tuple[NullStatefulPolicyState, Float[Array, " actions"]]:
        return state, self.policy.q_values(observation)


type AbstractQPolicy = AbstractStatefulQPolicy | AbstractStatelessQPolicy
