from __future__ import annotations

from typing import cast

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, Real

from lerax.distribution import (
    AbstractDistribution,
    Categorical,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)
from lerax.env import AbstractEnvLike
from lerax.model import MLP
from lerax.space import Box, Discrete

from .actor_critic import AbstractActorCriticPolicy


class MLPActorCriticPolicy[
    FeatureType: Array,
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractActorCriticPolicy[FeatureType, ActType, ObsType]):
    """
    Actor–critic policy with MLP components.
    """

    action_space: Box | Discrete
    observation_space: Box | Discrete

    feature_extractor: MLP
    value_model: MLP
    action_model: MLP
    log_std: Float[Array, " action_size"]

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
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

    def extract_features(
        self, state: eqx.nn.State, observation: ObsType
    ) -> tuple[eqx.nn.State, FeatureType]:
        """Extract features from an observation."""
        features = self.feature_extractor(jnp.ravel(observation))
        return state, cast(FeatureType, features)

    def action_dist_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[
        eqx.nn.State,
        AbstractDistribution[ActType],
    ]:
        """Return an action distribution from features."""
        action_mean = self.action_model(features)

        if isinstance(self.action_space, Discrete):
            action_dist: AbstractDistribution[ActType] = cast(
                AbstractDistribution[ActType],
                Categorical(logits=action_mean),
            )
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
            action_dist = cast(AbstractDistribution[ActType], base_dist)
        else:
            raise NotImplementedError(
                f"Action space {type(self.action_space)} not supported."
            )

        return state, action_dist

    def value_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        """Return a value from features."""
        value = self.value_model(features)
        return state, value

    def reset(self, state: eqx.nn.State) -> eqx.nn.State:
        """Reset the policy state."""
        return state
