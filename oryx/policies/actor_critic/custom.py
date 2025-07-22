import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key

from oryx.distributions import (
    AbstractSampleLogProbDistribution,
    AbstractTransformedDistribution,
    Categorical,
    SquashedMultivariateNormalDiag,
)
from oryx.env import AbstractEnvLike
from oryx.models import MLP, AbstractModel, AbstractStatefulModel, Flatten
from oryx.spaces import Box, Discrete

from .actor_critic import AbstractActorCriticPolicy


# TODO: Support other action spaces
class CustomActorCriticPolicy(
    AbstractActorCriticPolicy[Float, Float, Float], strict=True
):
    """
    Actor-critic policy with custom feature extractor, value model, and action model.

    If no feature extractor is provided, the flatten model will be used.
    If no value model is provided, a default MLP will be used.
    If no action model is provided, a default MLP will be used.

    Only Box and Discrete action spaces are supported currently.
    Only non-stochastic models are supported currently.
    """

    state_index: eqx.nn.StateIndex[None]

    feature_extractor: AbstractModel[[Float], Float]
    value_model: AbstractModel[[Float], Float[Array, ""]]
    action_model: AbstractModel[[Float], Float]
    log_std: Float[Array, " action_size"]

    env: AbstractEnvLike[Float, Float]

    def __init__(
        self,
        env: AbstractEnvLike[Float, Float],
        *,
        feature_extractor: AbstractModel[[Float], Float] | None = None,
        feature_size: int | None = None,
        value_model: AbstractModel[[Float], Float[Array, ""]] | None = None,
        action_model: AbstractModel[[Float], Float] | None = None,
        key: Key,
    ):
        self.env = env

        if feature_extractor is None:
            feature_extractor = Flatten()
            feature_size = int(jnp.prod(jnp.asarray(env.observation_space.shape)))
        elif feature_size is None:
            raise ValueError("Custom feature extractor must specify feature_size")

        self.feature_extractor = feature_extractor

        if value_model is None:
            key, value_model_key = jr.split(key, 2)
            value_model = MLP(
                in_size=feature_size,
                out_size="scalar",
                width_size=128,
                depth=4,
                key=value_model_key,
            )

        self.value_model = value_model

        if action_model is None:
            key, action_model_key = jr.split(key, 2)
            action_model = MLP(
                in_size=feature_size,
                out_size=int(jnp.prod(jnp.asarray(env.action_space.shape))),
                width_size=128,
                depth=4,
                key=action_model_key,
            )
            self.log_std = jnp.zeros(env.action_space.shape)

        self.action_model = action_model

        self.state_index = eqx.nn.StateIndex(None)

    def extract_features(
        self, state: eqx.nn.State, observation: Float
    ) -> tuple[eqx.nn.State, Float]:
        if isinstance(self.feature_extractor, AbstractStatefulModel):
            substate = state.substate(self.feature_extractor)
            substate, features = self.feature_extractor(substate, observation)
            state = state.update(substate)

        else:
            features = self.feature_extractor(observation)

        return state, features

    def action_dist_from_features(self, state: eqx.nn.State, features: Float) -> tuple[
        eqx.nn.State,
        AbstractSampleLogProbDistribution[Float]
        | AbstractTransformedDistribution[Float],
    ]:
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
        self, state: eqx.nn.State, features: Float
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        if isinstance(self.value_model, AbstractStatefulModel):
            substate = state.substate(self.value_model)
            # FIX: Typing not working here for some reason
            substate, value = self.value_model(substate, features)  # pyright: ignore
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
