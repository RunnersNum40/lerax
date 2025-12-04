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
    """
    Base class for stateless Deep Q-Network (DQN) policies.

    DQN policies utilize two Q-networks: a primary Q-network for action
    selection and a target Q-network for stable learning.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        q_network: The primary Q-network used for action selection.
        target_q_network: The target Q-network used for stable learning.
    """

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
    """
    Base class for stateful Deep Q-Network (DQN) policies.

    DQN policies utilize two Q-networks: a primary Q-network for action
    selection and a target Q-network for stable learning.

    In a stateful DQN policy the Q-networks should be identical in structure
    and differ only in their parameters. Due to this the state is shared.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        q_network: The primary Q-network used for action selection.
        target_q_network: The target Q-network used for stable learning.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    q_network: PolicyType
    target_q_network: PolicyType

    def __call__(
        self, state: StateType, observation: ObsType, *, key: Key | None = None
    ) -> tuple[StateType, Integer[Array, ""]]:
        """
        Return the next state and action for a given observation and state.

        Acts by selecting the action with the highest Q-value or random action
        based on epsilon-greedy exploration.

        Args:
            state: The current internal state of the policy.
            observation: The current observation from the environment.
            key: JAX PRNG key for stochastic action selection. If None, the
                action with the highest Q-value is always selected.

        Returns:
            Tuple of the next internal state and the selected action.
        """
        return self.q_network(state, observation, key=key)

    def q_values(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Array]:
        """
        Return Q-values for all actions given an observation and state.

        Args:
            state: The current internal state of the policy.
            observation: The current observation from the environment.

        Returns:
            Tuple of the next internal state and the Q-values for all actions.
        """
        return self.q_network.q_values(state, observation)

    def target_q_values(
        self, state: StateType, observation: ObsType
    ) -> tuple[StateType, Array]:
        """
        Return target Q-values for all actions given an observation and state.

        Args:
            state: The current internal state of the policy.
            observation: The current observation from the environment.

        Returns:
            Tuple of the next internal state and the target Q-values for all actions.
        """
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


type AbstractDQNPolicy = AbstractStatefulDQNPolicy | AbstractStatelessDQNPolicy
