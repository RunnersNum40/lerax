from __future__ import annotations

from typing import ClassVar

from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, Real

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.model import MLP
from lerax.space import AbstractSpace

from .base_actor_critic import AbstractStatelessActorCriticPolicy
from .utils import ActionHead


class MLPActorCriticPolicy[
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractStatelessActorCriticPolicy[ActType, ObsType]):
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

    action_space: AbstractSpace[ActType]
    observation_space: AbstractSpace[ObsType]

    encoder: MLP
    value_head: MLP
    action_head: ActionHead

    def __init__[StateType: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType],
        *,
        feature_size: int = 16,
        feature_width: int = 64,
        feature_depth: int = 2,
        value_width: int = 64,
        value_depth: int = 2,
        action_width: int = 64,
        action_depth: int = 2,
        log_std_init: float = 0.0,
        key: Key,
    ):
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        feat_key, val_key, act_key = jr.split(key, 3)

        self.encoder = MLP(
            in_size=self.observation_space.flat_size,
            out_size=feature_size,
            width_size=feature_width,
            depth=feature_depth,
            key=feat_key,
        )

        self.value_head = MLP(
            in_size=feature_size,
            out_size="scalar",
            width_size=value_width,
            depth=value_depth,
            key=val_key,
        )

        self.action_head = ActionHead(
            self.action_space,
            feature_size,
            action_width,
            action_depth,
            key=act_key,
            log_std_init=log_std_init,
        )

    def __call__(self, observation: ObsType, *, key: Key | None = None) -> ActType:
        features = self.encoder(self.observation_space.flatten_sample(observation))
        action_dist = self.action_head(features)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return action

    def action_and_value(
        self, observation: ObsType, *, key: Key
    ) -> tuple[ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """
        features = self.encoder(self.observation_space.flatten_sample(observation))
        value = self.value_head(features)

        action_dist = self.action_head(features)
        action, log_prob = action_dist.sample_and_log_prob(key)

        return action, value, log_prob.sum().squeeze()

    def value(self, observation: ObsType) -> Float[Array, ""]:
        """Get the value of an observation."""
        features = self.encoder(self.observation_space.flatten_sample(observation))
        return self.value_head(features)

    def evaluate_action(
        self, observation: ObsType, action: ActType
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Evaluate an action given an observation."""
        features = self.encoder(self.observation_space.flatten_sample(observation))
        action_dist = self.action_head(features)
        value = self.value_head(features)
        log_prob = action_dist.log_prob(action)

        try:
            entropy = action_dist.entropy().squeeze()
        except NotImplementedError:
            entropy = -log_prob.mean().squeeze()  # Fallback to negative log prob mean

        return value, log_prob.sum().squeeze(), entropy
