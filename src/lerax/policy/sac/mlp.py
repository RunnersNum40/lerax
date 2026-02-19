from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, PyTree, Real

from lerax.distribution import (
    AbstractDistribution,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Box

from .base_sac import AbstractSACPolicy

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class MLPSACPolicy[ObsType: PyTree[Real[Array, "..."]]](
    AbstractSACPolicy[None, Float[Array, " action_dim"], ObsType]
):
    """
    SAC policy with MLP components.

    Uses an MLP encoder to process observations, then separate linear heads
    for action mean and state-dependent log standard deviation. Produces
    squashed Gaussian distributions for bounded continuous actions.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        encoder: MLP to encode observations into features.
        mean_head: Linear layer producing action means.
        log_std_head: Linear layer producing state-dependent log standard deviations.
        scalar: Whether the action space is scalar.

    Args:
        env: The environment to create the policy for.
        feature_size: Size of the feature representation.
        width_size: Width of the hidden layers in the encoder.
        depth: Depth of the hidden layers in the encoder.
        key: JAX PRNG key for parameter initialization.
    """

    name: ClassVar[str] = "MLPSACPolicy"

    action_space: Box
    observation_space: AbstractSpace[ObsType, Any]

    encoder: eqx.nn.MLP
    mean_head: eqx.nn.Linear
    log_std_head: eqx.nn.Linear
    scalar: bool

    def __init__[S: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[S, Float[Array, " action_dim"], ObsType, None],
        *,
        feature_size: int = 256,
        width_size: int = 256,
        depth: int = 2,
        key: Key[Array, ""],
    ):
        assert isinstance(env.action_space, Box), "SAC requires a Box action space"

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.scalar = not env.action_space.shape

        encoder_key, mean_key, log_std_key = jr.split(key, 3)

        self.encoder = eqx.nn.MLP(
            in_size=self.observation_space.flat_size,
            out_size=feature_size,
            width_size=width_size,
            depth=depth,
            key=encoder_key,
        )

        action_size = max(int(self.action_space.flat_size), 1)
        out_size: int | str = "scalar" if self.scalar else action_size
        self.mean_head = eqx.nn.Linear(feature_size, out_size, key=mean_key)
        self.log_std_head = eqx.nn.Linear(feature_size, out_size, key=log_std_key)

    def reset(self, *, key: Key[Array, ""]) -> None:
        return None

    def _get_distribution(
        self, observation: ObsType
    ) -> SquashedNormal | SquashedMultivariateNormalDiag:
        """Compute the squashed Gaussian distribution from an observation."""
        features = self.encoder(self.observation_space.flatten_sample(observation))
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = jnp.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = jnp.exp(log_std)

        if self.scalar:
            return SquashedNormal(
                loc=mean,
                scale=std,
                high=self.action_space.high,
                low=self.action_space.low,
            )
        else:
            return SquashedMultivariateNormalDiag(
                loc=mean,
                scale_diag=std,
                high=self.action_space.high,
                low=self.action_space.low,
            )

    def __call__(
        self,
        state: None,
        observation: ObsType,
        *,
        key: Key[Array, ""] | None = None,
        action_mask: None = None,
    ) -> tuple[None, Float[Array, " action_dim"]]:
        dist = self._get_distribution(observation)
        if key is None:
            action = dist.mode()
        else:
            action = dist.sample(key)
        return None, action

    def action_distribution(
        self,
        state: None,
        observation: ObsType,
    ) -> tuple[None, AbstractDistribution[Float[Array, " action_dim"]]]:
        return None, self._get_distribution(observation)

    def action_and_log_prob(
        self,
        state: None,
        observation: ObsType,
        *,
        key: Key[Array, ""],
    ) -> tuple[None, Float[Array, " action_dim"], Float[Array, ""]]:
        dist = self._get_distribution(observation)
        action, log_prob = dist.sample_and_log_prob(key)
        return None, action, log_prob.sum().squeeze()
