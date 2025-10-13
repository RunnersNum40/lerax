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
from lerax.model import MLP, MLPNeuralCDE
from lerax.space import Box, Discrete

from .actor_critic import AbstractActorCriticPolicy


class NCDEActorCriticPolicy[
    FeatureType: Array,
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractActorCriticPolicy[FeatureType, ActType, ObsType]):
    """
    Actor–critic policy with an NCDE feature extractor.
    """

    action_space: Box | Discrete
    observation_space: Box | Discrete

    time_index: eqx.nn.StateIndex[Float[Array, ""]]

    feature_extractor: MLPNeuralCDE
    value_model: MLP
    action_model: MLP
    log_std: Float[Array, " action_size"]

    dt: float = eqx.field(static=True)

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        *,
        feature_size: int = 64,
        latent_size: int = 64,
        field_width: int = 64,
        field_depth: int = 2,
        initial_width: int = 64,
        initial_depth: int = 1,
        output_width: int = 64,
        output_depth: int = 1,
        state_size: int = 16,
        time_in_input: bool = False,
        dt: float = 1.0,
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

        fe_key, val_key, act_key = jr.split(key, 3)
        in_size = int(jnp.asarray(self.observation_space.flat_size))

        # TODO: Allow inference mode
        # Inference mode must be disabled when evaluating action values during
        # training but could be disabled during rollout. Currently there is no
        # way to know which mode we are in.
        self.feature_extractor = MLPNeuralCDE(
            in_size=in_size,
            out_size=feature_size,
            latent_size=latent_size,
            field_width=field_width,
            field_depth=field_depth,
            initial_width=initial_width,
            initial_depth=initial_depth,
            output_width=output_width,
            output_depth=output_depth,
            time_in_input=time_in_input,
            inference=False,
            state_size=state_size,
            key=fe_key,
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

        self.time_index = eqx.nn.StateIndex(jnp.asarray(0.0))
        self.dt = float(dt)

    def extract_features(
        self, state: eqx.nn.State, observation: ObsType
    ) -> tuple[eqx.nn.State, FeatureType]:
        t = state.get(self.time_index)
        t1 = t + self.dt

        obs_flat = jnp.ravel(observation)

        fe_state = state.substate(self.feature_extractor)
        fe_state, features = self.feature_extractor(fe_state, t1, obs_flat)
        state = state.update(fe_state)
        state = state.set(self.time_index, t1)

        return state, cast(FeatureType, features)

    def action_dist_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, AbstractDistribution[ActType]]:
        action_mean = self.action_model(features)

        if isinstance(self.action_space, Discrete):
            dist: AbstractDistribution[ActType] = cast(
                AbstractDistribution[ActType], Categorical(logits=action_mean)
            )
        elif isinstance(self.action_space, Box):
            if self.action_space.shape == ():
                base = SquashedNormal(loc=action_mean, scale=jnp.exp(self.log_std))
            else:
                base = SquashedMultivariateNormalDiag(
                    loc=action_mean, scale_diag=jnp.exp(self.log_std)
                )
            dist = cast(AbstractDistribution[ActType], base)
        else:
            raise NotImplementedError(
                f"Action space {type(self.action_space)} not supported."
            )

        return state, dist

    def value_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        value = self.value_model(features)
        return state, value

    def reset(self, state: eqx.nn.State) -> eqx.nn.State:
        fe_state = state.substate(self.feature_extractor)
        fe_state = self.feature_extractor.reset(fe_state)
        state = state.update(fe_state)
        state = state.set(self.time_index, jnp.asarray(0.0))
        return state
