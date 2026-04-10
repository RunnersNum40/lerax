from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, PyTree, Real

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Box

from .base_deterministic import AbstractDeterministicPolicy


class MLPDeterministicPolicy[ObsType: PyTree[Real[Array, "..."]]](
    AbstractDeterministicPolicy[None, Float[Array, " action_dim"], ObsType]
):
    """
    Deterministic MLP policy for continuous action spaces.

    Produces deterministic actions via an MLP with tanh output scaled to
    action bounds. Used by DDPG and TD3 — exploration noise is added by
    the algorithm, not the policy.

    When ``key`` is ``None``, returns the deterministic action.
    When ``key`` is provided, adds Gaussian exploration noise scaled by
    ``exploration_noise``.

    Attributes:
        name: Name of the policy class.
        action_space: The (continuous) action space.
        observation_space: The observation space.
        network: MLP producing raw action outputs.
        action_scale: Half the range of the action space.
        action_bias: Midpoint of the action space.
        exploration_noise: Standard deviation of exploration noise
            (relative to action scale).

    Args:
        env: The environment to create the policy for.
        width_size: Width of hidden layers.
        depth: Number of hidden layers.
        exploration_noise: Exploration noise scale.
        key: JAX PRNG key for parameter initialization.
    """

    name: ClassVar[str] = "MLPDeterministicPolicy"

    action_space: Box
    observation_space: AbstractSpace[ObsType, Any]

    network: eqx.nn.MLP
    action_scale: Float[Array, " action_dim"]
    action_bias: Float[Array, " action_dim"]
    scalar: bool
    exploration_noise: float

    def __init__[S: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[S, Float[Array, " action_dim"], ObsType, None],
        *,
        width_size: int = 256,
        depth: int = 2,
        exploration_noise: float = 0.1,
        key: Key[Array, ""],
    ):
        assert isinstance(env.action_space, Box), (
            "Deterministic policy requires a Box action space"
        )

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.exploration_noise = exploration_noise

        self.scalar = not env.action_space.shape
        action_size = max(int(env.action_space.flat_size), 1)
        out_size: int | str = "scalar" if self.scalar else action_size
        self.network = eqx.nn.MLP(
            in_size=env.observation_space.flat_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jnp.tanh,
            final_activation=jnp.tanh,
            key=key,
        )

        self.action_scale = (env.action_space.high - env.action_space.low) / 2.0
        self.action_bias = (env.action_space.high + env.action_space.low) / 2.0

    def reset(self, *, key: Key[Array, ""]) -> None:
        return None

    def __call__(
        self,
        state: None,
        observation: ObsType,
        *,
        key: Key[Array, ""] | None = None,
        action_mask: None = None,
    ) -> tuple[None, Float[Array, " action_dim"]]:
        flat_obs = self.observation_space.flatten_sample(observation)
        raw_action = self.network(flat_obs)
        action = raw_action * self.action_scale + self.action_bias

        if key is not None:
            noise = (
                jr.normal(key, action.shape)
                * self.action_scale
                * self.exploration_noise
            )
            action = jnp.clip(
                action + noise,
                self.action_space.low,
                self.action_space.high,
            )

        return None, action
