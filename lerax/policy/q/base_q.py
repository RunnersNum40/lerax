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
    """
    Base class for stateless Q-learning policies.

    Q-learning policies map observations to actions by selecting the action
    with the highest Q-value, with epsilon-greedy exploration.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        epsilon: The epsilon value for epsilon-greedy action selection.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    epsilon: eqx.AbstractVar[float]

    @abstractmethod
    def q_values(self, observation: ObsType) -> Float[Array, " actions"]:
        """
        Return Q-values for all actions given an observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            Q-values for all actions.
        """

    def into_stateful[SelfType: AbstractStatelessQPolicy](
        self: SelfType,
    ) -> QStatefulWrapper[SelfType, ObsType]:
        return QStatefulWrapper(self)

    def __call__(
        self, observation: ObsType, *, key: Array | None = None
    ) -> Integer[Array, ""]:
        """
        Return the action for a given observation.

        Args:
            observation: The current observation from the environment.
            key: JAX PRNG key for stochastic action selection. If None, the
                action with the highest Q-value is always selected.

        Returns:
            The selected action.
        """
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
    """
    Base class for stateful Q-learning policies.

    Q-learning policies map observations and internal states to actions by
    selecting the action with the highest Q-value, with epsilon-greedy
    exploration.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        epsilon: The epsilon value for epsilon-greedy action selection.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    epsilon: eqx.AbstractVar[float]

    @abstractmethod
    def q_values(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Float[Array, " actions"]]:
        """
        Return Q-values for all actions given an observation and state.

        Args:
            state: The current internal state of the policy.
            observation: The current observation from the environment.

        Returns:
            A tuple of the next internal state and the Q-values for all actions.
        """

    def __call__(
        self, state: StateType, observation: ObsType, *, key: Array | None = None
    ) -> tuple[StateType, Integer[Array, ""]]:
        """
        Return the next state and action for a given observation and state.

        Uses epsilon-greedy action selection.

        Args:
            state: The current internal state of the policy.
            observation: The current observation from the environment.
            key: JAX PRNG key for stochastic action selection. If None, the
                action with the highest Q-value is always selected.

        Returns:
            A tuple of the next internal state and the selected action.
        """
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
