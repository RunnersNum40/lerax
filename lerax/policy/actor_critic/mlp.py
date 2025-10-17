from __future__ import annotations

from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, Real

from lerax.distribution import (
    AbstractDistribution,
    Categorical,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.model import MLP
from lerax.space import Box, Discrete

from .base_actor_critic import AbstractStatelessActorCriticPolicy


class MLPActorCriticPolicy[
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractStatelessActorCriticPolicy[ActType, ObsType]):
    """
    Actor–critic policy with MLP components.
    """

    action_space: Box | Discrete
    observation_space: Box | Discrete

    feature_extractor: MLP
    value_model: MLP
    action_model: MLP
    log_std: Float[Array, " action_size"]

    def __init__[StateType: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType],
        *,
        feature_size: int = 64,
        feature_width: int = 64,
        feature_depth: int = 0,
        value_width: int = 64,
        value_depth: int = 2,
        action_width: int = 64,
        action_depth: int = 2,
        log_std_init: float = 0.0,
        key: Key,
    ):
        if isinstance(env.action_space, Discrete):
            act_size = int(env.action_space.n)
            self.log_std = jnp.array([], dtype=float)
        elif isinstance(env.action_space, Box):
            if env.action_space.shape:
                act_size = int(jnp.prod(jnp.asarray(env.action_space.shape)))
                self.log_std = jnp.full((act_size,), log_std_init, dtype=float)
            else:
                act_size = "scalar"
                self.log_std = jnp.array(log_std_init, dtype=float)
        else:
            raise NotImplementedError(
                f"Action space {type(env.action_space)} not supported."
            )

        if not isinstance(env.observation_space, (Discrete, Box)):
            raise NotImplementedError(
                f"Observation space {type(env.observation_space)} not supported."
            )

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        feat_key, val_key, act_key = jr.split(key, 3)

        self.feature_extractor = MLP(
            in_size=int(jnp.array(self.observation_space.flat_size)),
            out_size=feature_size,
            width_size=feature_width,
            depth=feature_depth,
            key=feat_key,
        )

        self.value_model = MLP(
            in_size=feature_size,
            out_size="scalar",
            width_size=value_width,
            depth=value_depth,
            key=val_key,
        )

        self.action_model = MLP(
            in_size=feature_size,
            out_size=act_size,
            width_size=action_width,
            depth=action_depth,
            key=act_key,
        )

    def extract_features(self, observation: ObsType) -> Float[Array, " feature_size"]:
        """Extract features from an observation."""
        return self.feature_extractor(jnp.ravel(observation))

    def action_dist_from_features(
        self, features: Float[Array, " feature_size"]
    ) -> AbstractDistribution[ActType]:
        """Return an action distribution from features."""
        action_mean = self.action_model(features)

        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(logits=action_mean)
        elif isinstance(self.action_space, Box):
            if self.action_space.shape == ():
                base_dist = SquashedNormal(
                    loc=action_mean,
                    scale=jnp.exp(self.log_std),
                )
            else:
                base_dist = SquashedMultivariateNormalDiag(
                    loc=action_mean,
                    scale_diag=jnp.exp(self.log_std),
                )
            action_dist = base_dist
        else:
            raise NotImplementedError(
                f"Action space {type(self.action_space)} not supported."
            )

        return action_dist

    def value_from_features(
        self, features: Float[Array, " feature_size"]
    ) -> Float[Array, ""]:
        """Return a value from features."""
        return self.value_model(features)

    def action_and_value(
        self, observation: ObsType, *, key: Key | None = None
    ) -> tuple[ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """
        features = self.extract_features(observation)
        action_dist = self.action_dist_from_features(features)
        value = self.value_from_features(features)

        if key is None:
            action = action_dist.mode()
            log_prob = action_dist.log_prob(action)
        else:
            action, log_prob = action_dist.sample_and_log_prob(key)

        return action, value, log_prob.sum().squeeze()

    def value(self, observation: ObsType) -> Float[Array, ""]:
        """Get the value of an observation."""
        features = self.extract_features(observation)
        return self.value_from_features(features)

    def evaluate_action(
        self, observation: ObsType, action: ActType
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Evaluate an action given an observation."""
        features = self.extract_features(observation)
        action_dist = self.action_dist_from_features(features)
        value = self.value_from_features(features)
        log_prob = action_dist.log_prob(action)

        try:
            entropy = action_dist.entropy().squeeze()
        except NotImplementedError:
            entropy = -log_prob.mean().squeeze()  # Fallback to negative log prob mean

        return value, log_prob.sum().squeeze(), entropy
