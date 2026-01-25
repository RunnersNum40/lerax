from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jaxtyping import Array, Key

from lerax.space import AbstractSpace
from lerax.utils import Serializable


class AbstractPolicyState(eqx.Module):
    """
    Base class for policy internal states.
    """

    pass


class AbstractPolicy[StateType: AbstractPolicyState | None, ActType, ObsType, MaskType](
    Serializable
):
    """
    Base class for policies.

    Policies map observations and internal states to actions and new internal states.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType, MaskType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType, Any]]

    @abstractmethod
    def __call__(
        self,
        state: StateType,
        observation: ObsType,
        *,
        key: Key[Array, ""] | None = None,
        action_mask: MaskType | None = None,
    ) -> tuple[StateType, ActType]:
        """
        Return the next action and new internal state given the current
        observation and internal state.

        A key can be provided for stochastic policies. If no key is provided,
        the policy should behave deterministically.

        Args:
            state: The current internal state of the policy.
            observation: The current observation.
            key: An optional JAX random key for stochastic policies.
            action_mask: An optional action mask.

        Returns:
            The new internal state and the action to take.
        """
        pass

    @abstractmethod
    def reset(self, *, key: Key[Array, ""]) -> StateType:
        """
        Return an initial internal state for the policy.

        Args:
            key: A JAX random key for initializing the state.

        Returns:
            An initial internal state for the policy.
        """
        pass
