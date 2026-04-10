from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer, Key, PyTree, Real

from lerax.distribution import AbstractDistribution, Categorical
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Discrete

from .base_sac import AbstractSACPolicy


class MLPDiscreteSACPolicy[ObsType: PyTree[Real[Array, "..."]]](
    AbstractSACPolicy[None, Integer[Array, ""], ObsType]
):
    """
    Discrete SAC policy with MLP backbone.

    Outputs a categorical distribution over discrete actions. Used by
    SAC-Discrete where entropy regularization is applied to the
    categorical action distribution.

    Attributes:
        name: Name of the policy class.
        action_space: The discrete action space.
        observation_space: The observation space.
        network: MLP producing action logits.

    Args:
        env: The environment to create the policy for.
        width_size: Width of hidden layers.
        depth: Number of hidden layers.
        key: JAX PRNG key for parameter initialization.
    """

    name: ClassVar[str] = "MLPDiscreteSACPolicy"

    action_space: Discrete
    observation_space: AbstractSpace[ObsType, Any]

    network: eqx.nn.MLP

    def __init__[S: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[S, Integer[Array, ""], ObsType, Any],
        *,
        width_size: int = 256,
        depth: int = 2,
        key: Key[Array, ""],
    ):
        assert isinstance(env.action_space, Discrete), (
            "Discrete SAC policy requires a Discrete action space"
        )

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.network = eqx.nn.MLP(
            in_size=env.observation_space.flat_size,
            out_size=env.action_space.n,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def reset(self, *, key: Key[Array, ""]) -> None:
        return None

    def _get_logits(self, observation: ObsType) -> Float[Array, " num_actions"]:
        """Compute action logits from an observation."""
        return self.network(self.observation_space.flatten_sample(observation))

    def __call__(
        self,
        state: None,
        observation: ObsType,
        *,
        key: Key[Array, ""] | None = None,
        action_mask: None = None,
    ) -> tuple[None, Integer[Array, ""]]:
        logits = self._get_logits(observation)
        if key is None:
            action = jnp.argmax(logits)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample(key)
        return None, action

    def action_distribution(
        self,
        state: None,
        observation: ObsType,
    ) -> tuple[None, AbstractDistribution[Integer[Array, ""]]]:
        logits = self._get_logits(observation)
        return None, Categorical(logits=logits)

    def action_and_log_prob(
        self,
        state: None,
        observation: ObsType,
        *,
        key: Key[Array, ""],
    ) -> tuple[None, Integer[Array, ""], Float[Array, ""]]:
        logits = self._get_logits(observation)
        dist = Categorical(logits=logits)
        action = dist.sample(key)
        log_prob = dist.log_prob(action)
        return None, action, log_prob

    def action_probs_and_log_probs(
        self,
        state: None,
        observation: ObsType,
    ) -> tuple[None, Float[Array, " num_actions"], Float[Array, " num_actions"]]:
        """
        Return action probabilities and per-action log probabilities.

        Used by SAC-Discrete for computing the expected entropy and
        expected Q-value over all actions.

        Args:
            state: The current internal state of the policy (unused).
            observation: The current observation.

        Returns:
            The internal state (None), action probabilities, and
            per-action log probabilities.
        """
        logits = self._get_logits(observation)
        probs = jax.nn.softmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        return None, probs, log_probs
