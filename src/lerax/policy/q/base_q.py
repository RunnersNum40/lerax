from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jax import lax
from jax import random as jr
from jaxtyping import Array, Bool, Float, Integer

from lerax.distribution import Categorical
from lerax.space import AbstractSpace, Discrete

from ..base_policy import AbstractPolicy, AbstractPolicyState


class AbstractQPolicy[StateType: AbstractPolicyState | None, ObsType](
    AbstractPolicy[StateType, Integer[Array, ""], ObsType, Bool[Array, " actions"]],
):
    """
    Base class for stateful epsilon-greedy Q-learning policies.

    Epsilon-greedy policies select a random action with probability *epsilon*
    and the action with the highest Q-value with probability 1-*epsilon*.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        epsilon: The epsilon value for epsilon-greedy action selection.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[Discrete]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType, Any]]

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
        self,
        state: StateType,
        observation: ObsType,
        *,
        action_mask: Bool[Array, " actions"] | None = None,
        key: Array | None = None,
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
        dist = Categorical(logits=q_vals)
        if action_mask is not None:
            dist = dist.mask(action_mask)

        if key is None or self.epsilon <= 0.0:
            return state, dist.mode()
        else:
            epsilon_key, action_key = jr.split(key, 2)
            action = lax.cond(
                jr.uniform(epsilon_key, shape=()) < self.epsilon,
                lambda: dist.sample(action_key),
                lambda: dist.mode(),
            )
            return state, action
