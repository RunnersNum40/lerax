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
from oryx.space.base_space import AbstractSpace

from .actor_critic import AbstractActorCriticPolicy


class CustomActorCriticPolicy[
    FeatureType: Array,
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractActorCriticPolicy[FeatureType, ActType, ObsType]):
    """
    Actor–critic policy with pluggable components.

    Defaults:
      - Feature extractor: Flatten
      - Value model: MLP(feature_size → scalar)
      - Action model:
          * Discrete: MLP(feature_size → n_actions) with Categorical
          * Box: MLP(feature_size → action_dim) with Normal + squashing
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

        self.state_index = eqx.nn.StateIndex(None)

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space

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

        if isinstance(self.env.action_space, Box):
            if self.env.action_space.shape == ():
                dist = SquashedNormal(
                    loc=action_param,
                    scale=jnp.exp(self.log_std),
                    high=self.env.action_space.high,
                    low=self.env.action_space.low,
                )
            else:
                dist = SquashedMultivariateNormalDiag(
                    loc=action_param,
                    scale_diag=jnp.exp(self.log_std),
                    high=self.env.action_space.high,
                    low=self.env.action_space.low,
                )
        elif isinstance(self.env.action_space, Discrete):
            dist = Categorical(logits=action_param)
        else:
            raise NotImplementedError(
                f"Unsupported action space {type(self.env.action_space)}."
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
