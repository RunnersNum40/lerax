from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, PyTree, Real

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace

from ..actor import ActionLayer
from .base_actor_critic import AbstractActorCriticPolicy


class MLPActorCriticPolicy[
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: PyTree[Real[Array, "..."]],
    MaskType,
](AbstractActorCriticPolicy[None, ActType, ObsType, MaskType]):
    """
    Actor–critic policy with MLP components.

    Uses an MLP to encode observations into features, then separate MLPs to
    produce action distributions and value estimates from those features.

    Action distributions are produced by mapping action head outputs to the
    parameters of the appropriate distribution for the action space.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        encoder: MLP to encode observations into features.
        value_head: MLP to produce value estimates from features.
        action_head: MLP to produce action distributions from features.

    Args:
        env: The environment to create the policy for.
        feature_size: Size of the feature representation.
        feature_width: Width of the hidden layers in the feature encoder.
        feature_depth: Depth of the hidden layers in the feature encoder.
        value_width: Width of the hidden layers in the value head.
        value_depth: Depth of the hidden layers in the value head.
        action_width: Width of the hidden layers in the action head.
        action_depth: Depth of the hidden layers in the action head.
        log_std_init: Initial log standard deviation for continuous action spaces.
        key: JAX PRNG key for parameter initialization.
    """

    name: ClassVar[str] = "MLPActorCriticPolicy"

    action_space: AbstractSpace[ActType, MaskType]
    observation_space: AbstractSpace[ObsType, Any]

    encoder: eqx.nn.MLP
    value_head: eqx.nn.MLP
    action_head: ActionLayer

    def __init__[S: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[S, ActType, ObsType, MaskType],
        *,
        feature_size: int = 16,
        feature_width: int = 64,
        feature_depth: int = 2,
        value_width: int = 64,
        value_depth: int = 2,
        action_width: int = 64,
        action_depth: int = 2,
        log_std_init: float = 0.0,
        key: Key[Array, ""],
    ):
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        feat_key, val_key, act_key = jr.split(key, 3)

        self.encoder = eqx.nn.MLP(
            in_size=self.observation_space.flat_size,
            out_size=feature_size,
            width_size=feature_width,
            depth=feature_depth,
            key=feat_key,
        )

        self.value_head = eqx.nn.MLP(
            in_size=feature_size,
            out_size="scalar",
            width_size=value_width,
            depth=value_depth,
            key=val_key,
        )

        self.action_head = ActionLayer(
            self.action_space,
            feature_size,
            action_width,
            action_depth,
            key=act_key,
            log_std_init=log_std_init,
        )

    def reset(self, *, key: Key[Array, ""]) -> None:
        return None

    def __call__(
        self,
        state: None,
        observation: ObsType,
        *,
        key: Key[Array, ""] | None = None,
        action_mask: MaskType | None = None,
    ) -> tuple[None, ActType]:
        features = self.encoder(self.observation_space.flatten_sample(observation))
        action_dist = self.action_head(features, action_mask=action_mask)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return None, action

    def action_and_value(
        self,
        state: None,
        observation: ObsType,
        *,
        key: Key[Array, ""],
        action_mask: MaskType | None = None,
    ) -> tuple[None, ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """
        features = self.encoder(self.observation_space.flatten_sample(observation))
        value = self.value_head(features)
        action_dist = self.action_head(features, action_mask=action_mask)
        action, log_prob = action_dist.sample_and_log_prob(key)

        return None, action, value, log_prob.sum().squeeze()

    def value(self, state: None, observation: ObsType) -> tuple[None, Float[Array, ""]]:
        """Get the value of an observation."""
        features = self.encoder(self.observation_space.flatten_sample(observation))
        return None, self.value_head(features)

    def evaluate_action(
        self,
        state: None,
        observation: ObsType,
        action: ActType,
        *,
        action_mask: MaskType | None = None,
    ) -> tuple[None, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Evaluate an action given an observation."""
        features = self.encoder(self.observation_space.flatten_sample(observation))
        action_dist = self.action_head(features, action_mask=action_mask)
        value = self.value_head(features)
        log_prob = action_dist.log_prob(action)

        try:
            entropy = action_dist.entropy().squeeze()
        except NotImplementedError:
            entropy = -log_prob.mean().squeeze()  # Fallback to negative log prob mean

        return None, value, log_prob.sum().squeeze(), entropy
