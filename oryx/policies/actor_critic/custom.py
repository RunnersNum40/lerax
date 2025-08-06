from __future__ import annotations

from typing import cast

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, Real

from oryx.distributions import (
    AbstractDistribution,
    Categorical,
    SquashedMultivariateNormalDiag,
)
from oryx.env import AbstractEnvLike
from oryx.models import MLP, AbstractModel, AbstractStatefulModel, Flatten
from oryx.spaces import Box, Discrete
from oryx.spaces.base_space import AbstractSpace

from .actor_critic import AbstractActorCriticPolicy


# TODO: Support other action spaces
class CustomActorCriticPolicy[
    FeatureType: Array,
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractActorCriticPolicy[FeatureType, ActType, ObsType]):
    """
    Actor-critic policy with custom feature extractor, value model, and action model.

    If no feature extractor is provided, the flatten model will be used.
    If no value model is provided, a default MLP will be used.
    If no action model is provided, a default MLP will be used.

    Only Box and Discrete action spaces are supported currently.
    Only non-stochastic models are supported currently.
    """

    state_index: eqx.nn.StateIndex[None]

    feature_extractor: (
        AbstractModel[[ObsType], FeatureType]
        | AbstractStatefulModel[[ObsType], FeatureType]
    )
    value_model: (
        AbstractModel[[FeatureType], Float[Array, ""]]
        | AbstractStatefulModel[[FeatureType], Float[Array, ""]]
    )
    action_model: (
        AbstractModel[[FeatureType], ActType]
        | AbstractStatefulModel[[FeatureType], ActType]
    )
    log_std: Float[Array, " action_size"]

    env: AbstractEnvLike[ActType, ObsType]

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        *,
        feature_extractor: (
            AbstractModel[[ObsType], FeatureType]
            | AbstractStatefulModel[[ObsType], FeatureType]
            | None
        ) = None,
        feature_size: int | None = None,
        value_model: (
            AbstractModel[[FeatureType], Float[Array, ""]]
            | AbstractStatefulModel[[FeatureType], Float[Array, ""]]
            | None
        ) = None,
        action_model: (
            AbstractModel[[FeatureType], ActType]
            | AbstractStatefulModel[[FeatureType], ActType]
            | None
        ) = None,
        key: Key,
    ):
        self.env = env

        # TODO: Maybe cast is not needed here
        if feature_extractor is None:
            self.feature_extractor = cast(
                AbstractModel[[ObsType], FeatureType], Flatten()
            )
            feature_size = int(jnp.prod(jnp.asarray(env.observation_space.shape)))
        elif feature_size is None:
            raise ValueError("Custom feature extractor must specify feature_size")
        else:
            self.feature_extractor = feature_extractor

        if value_model is None:
            key, value_model_key = jr.split(key, 2)
            self.value_model = cast(
                AbstractModel[[FeatureType], Float[Array, ""]],
                MLP(
                    in_size=feature_size,
                    out_size="scalar",
                    width_size=128,
                    depth=4,
                    key=value_model_key,
                ),
            )
        else:
            self.value_model = value_model

        if action_model is None:
            key, action_model_key = jr.split(key, 2)
            self.action_model = cast(
                AbstractModel[[FeatureType], ActType],
                MLP(
                    in_size=feature_size,
                    out_size=int(jnp.prod(jnp.asarray(env.action_space.shape))),
                    width_size=128,
                    depth=4,
                    key=action_model_key,
                ),
            )
            self.log_std = jnp.zeros(env.action_space.shape)
        else:
            self.action_model = action_model

        self.state_index = eqx.nn.StateIndex(None)

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space

    def extract_features(
        self, state: eqx.nn.State, observation: ObsType
    ) -> tuple[eqx.nn.State, FeatureType]:
        if isinstance(self.feature_extractor, AbstractStatefulModel):
            substate = state.substate(self.feature_extractor)
            substate, features = self.feature_extractor(substate, observation)
            state = state.update(substate)

        else:
            features = self.feature_extractor(observation)

        return state, features

    def action_dist_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, AbstractDistribution[ActType]]:
        if isinstance(self.action_model, AbstractStatefulModel):
            substate = state.substate(self.action_model)
            substate, action = self.action_model(substate, features)
            state = state.update(substate)

        else:
            action = self.action_model(features)

        if isinstance(self.env.action_space, Box):
            action_distribution = SquashedMultivariateNormalDiag(
                loc=action,
                scale_diag=jnp.exp(self.log_std),
                high=self.env.action_space.high,
                low=self.env.action_space.low,
            )
        elif isinstance(self.env.action_space, Discrete):
            action_distribution = Categorical(logits=action)
        else:
            raise NotImplementedError(
                f"Action space {self.env.action_space} not supported"
            )

        return state, action_distribution

    def value_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        if isinstance(self.value_model, AbstractStatefulModel):
            substate = state.substate(self.value_model)
            substate, value = self.value_model(substate, features)
            state = state.update(substate)

        else:
            value = self.value_model(features)

        return state, value

    def reset(self, state: eqx.nn.State) -> eqx.nn.State:
        if isinstance(self.feature_extractor, AbstractStatefulModel):
            substate = state.substate(self.feature_extractor)
            substate = self.feature_extractor.reset(substate)
            state = state.update(substate)

        if isinstance(self.value_model, AbstractStatefulModel):
            substate = state.substate(self.value_model)
            substate = self.value_model.reset(substate)
            state = state.update(substate)

        if isinstance(self.action_model, AbstractStatefulModel):
            substate = state.substate(self.action_model)
            substate = self.action_model.reset(substate)
            state = state.update(substate)

        return state
