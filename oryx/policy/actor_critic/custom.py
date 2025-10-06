from __future__ import annotations

from typing import cast

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, Real

from oryx.distribution import (
    AbstractDistribution,
    Categorical,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)
from oryx.env import AbstractEnvLike
from oryx.model import MLP, AbstractModel, AbstractStatefulModel, Flatten
from oryx.space import Box, Discrete

from .actor_critic import AbstractActorCriticPolicy


class CustomActorCriticPolicy[
    FeatureType: Array,
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractActorCriticPolicy[FeatureType, ActType, ObsType]):
    """
    Actor–critic policy with pluggable components.
    """

    action_space: Box | Discrete
    observation_space: Box | Discrete

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
        if not isinstance(env.action_space, (Box, Discrete)):
            raise NotImplementedError(
                f"Unsupported action space {type(env.action_space)}."
                "Only Box and Discrete are supported."
            )
        if not isinstance(env.observation_space, (Box, Discrete)):
            raise NotImplementedError(
                f"Unsupported action space {type(env.observation_space)}."
                "Only Box and Discrete are supported."
            )

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        if feature_extractor is None:
            self.feature_extractor = cast(
                AbstractModel[[ObsType], FeatureType], Flatten()
            )
            feature_size = int(jnp.prod(jnp.asarray(env.observation_space.shape)))
        else:
            if feature_size is None:
                raise ValueError("Custom feature extractor requires feature_size.")
            self.feature_extractor = feature_extractor

        if value_model is None:
            key, vm_key = jr.split(key)
            self.value_model = cast(
                AbstractModel[[FeatureType], Float[Array, ""]],
                MLP(
                    in_size=feature_size,
                    out_size="scalar",
                    width_size=128,
                    depth=4,
                    key=vm_key,
                ),
            )
        else:
            self.value_model = value_model

        self.log_std = jnp.zeros(
            getattr(env.action_space, "shape", ()), dtype=jnp.float32
        )
        if action_model is None:
            key, am_key = jr.split(key)
            if isinstance(env.action_space, Discrete):
                out_size = int(env.action_space.n)
            elif isinstance(env.action_space, Box):
                out_size = (
                    "scalar"
                    if env.action_space.shape == ()
                    else int(jnp.prod(jnp.asarray(env.action_space.shape)))
                )
            else:
                raise NotImplementedError(
                    f"Unsupported action space {type(env.action_space)}."
                )
            self.action_model = cast(
                AbstractModel[[FeatureType], ActType],
                MLP(
                    in_size=feature_size,
                    out_size=out_size,
                    width_size=128,
                    depth=4,
                    key=am_key,
                ),
            )
        else:
            self.action_model = action_model

    @staticmethod
    def _apply_model[T, X](
        state: eqx.nn.State,
        model: AbstractModel[[X], T] | AbstractStatefulModel[[X], T],
        x: X,
    ) -> tuple[eqx.nn.State, T]:
        if isinstance(model, AbstractStatefulModel):
            sub = state.substate(model)
            sub, out = model(sub, x)
            state = state.update(sub)
            return state, out
        else:
            return state, model(x)

    def extract_features(
        self, state: eqx.nn.State, observation: ObsType
    ) -> tuple[eqx.nn.State, FeatureType]:
        return self._apply_model(state, self.feature_extractor, observation)

    def action_dist_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, AbstractDistribution[ActType]]:
        state, action_param = self._apply_model(state, self.action_model, features)

        if isinstance(self.action_space, Box):
            if self.action_space.shape == ():
                dist = SquashedNormal(
                    loc=action_param,
                    scale=jnp.exp(self.log_std),
                    high=self.action_space.high,
                    low=self.action_space.low,
                )
            else:
                dist = SquashedMultivariateNormalDiag(
                    loc=action_param,
                    scale_diag=jnp.exp(self.log_std),
                    high=self.action_space.high,
                    low=self.action_space.low,
                )
        elif isinstance(self.action_space, Discrete):
            dist = Categorical(logits=action_param)
        else:
            raise NotImplementedError(
                f"Unsupported action space {type(self.action_space)}."
            )

        return state, dist

    def value_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        return self._apply_model(state, self.value_model, features)

    def reset(self, state: eqx.nn.State) -> eqx.nn.State:
        if isinstance(self.feature_extractor, AbstractStatefulModel):
            sub = state.substate(self.feature_extractor)
            sub = self.feature_extractor.reset(sub)
            state = state.update(sub)
        if isinstance(self.value_model, AbstractStatefulModel):
            sub = state.substate(self.value_model)
            sub = self.value_model.reset(sub)
            state = state.update(sub)
        if isinstance(self.action_model, AbstractStatefulModel):
            sub = state.substate(self.action_model)
            sub = self.action_model.reset(sub)
            state = state.update(sub)
        return state
