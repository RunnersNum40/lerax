from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float, Key

from lerax.distribution import AbstractDistribution
from lerax.space import AbstractSpace

from ..base_policy import AbstractPolicy, AbstractPolicyState


class AbstractSACPolicy[
    StateType: AbstractPolicyState | None,
    ActType,
    ObsType,
](AbstractPolicy[StateType, ActType, ObsType, None]):
    """
    Base class for SAC policies.

    SAC policies produce squashed Gaussian action distributions for
    continuous action spaces. Unlike actor-critic policies, SAC policies
    do not include a value function — SAC uses separate Q-networks.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType, None]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType, Any]]

    @abstractmethod
    def action_distribution(
        self,
        state: StateType,
        observation: ObsType,
    ) -> tuple[StateType, AbstractDistribution[ActType]]:
        """
        Return the action distribution for a given observation.

        Args:
            state: The current internal state of the policy.
            observation: The current observation.

        Returns:
            The new internal state and the action distribution.
        """

    @abstractmethod
    def action_and_log_prob(
        self,
        state: StateType,
        observation: ObsType,
        *,
        key: Key[Array, ""],
    ) -> tuple[StateType, ActType, Float[Array, ""]]:
        """
        Sample an action and return its log probability.

        Args:
            state: The current internal state of the policy.
            observation: The current observation.
            key: A JAX PRNG key for sampling.

        Returns:
            The new internal state, the sampled action, and its log probability.
        """
