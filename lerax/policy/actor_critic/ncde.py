from __future__ import annotations

from typing import ClassVar

import diffrax
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
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.model import MLP, MLPNeuralCDE, NCDEState
from lerax.policy.actor_critic.base_actor_critic import (
    AbstractPolicyState,
    AbstractStatefulActorCriticPolicy,
)
from lerax.space import Box, Discrete


class NCDEPolicyState(AbstractPolicyState):
    t: Float[Array, ""]
    cde: NCDEState

    def __init__(self, *, t: Float[Array, ""] = jnp.array(0.0), cde: NCDEState):
        self.t = jnp.asarray(t, dtype=float)
        self.cde = cde


class NCDEActorCriticPolicy[
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
](AbstractStatefulActorCriticPolicy[NCDEPolicyState, ActType, ObsType]):
    """
    Actor–critic with a shared MLPNeuralCDE encoder and MLP heads.
    """

    name: ClassVar[str] = "NCDEActorCriticPolicy"

    action_space: Box | Discrete
    observation_space: Box | Discrete

    encoder: MLPNeuralCDE
    value_model: MLP
    action_head: MLP
    log_std: Float[Array, " action_size"]

    dt: float = eqx.field(static=True)

    def __init__[StateType: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType],
        *,
        solver: diffrax.AbstractSolver | None = None,
        feature_size: int = 4,
        latent_size: int = 4,
        field_width: int = 8,
        field_depth: int = 1,
        initial_width: int = 16,
        initial_depth: int = 1,
        output_width: int = 16,
        output_depth: int = 1,
        value_width: int = 16,
        value_depth: int = 1,
        action_width: int = 16,
        action_depth: int = 1,
        history_length: int = 4,
        dt: float = 1.0,
        log_std_init: float = 0.0,
        key: Key,
    ):
        if not isinstance(env.observation_space, (Box, Discrete)):
            raise NotImplementedError(
                f"Observation space {type(env.observation_space)} not supported."
            )
        if not isinstance(env.action_space, (Box, Discrete)):
            raise NotImplementedError(
                f"Action space {type(env.action_space)} not supported."
            )

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.dt = float(dt)

        obs_flat = int(jnp.asarray(env.observation_space.flat_size))
        if isinstance(env.action_space, Discrete):
            act_size: int | str = int(env.action_space.n)
            self.log_std = jnp.array([], dtype=float)
        else:
            if env.action_space.shape:
                act_size = int(jnp.prod(jnp.asarray(env.action_space.shape)))
                self.log_std = jnp.full((act_size,), log_std_init, dtype=float)
            else:
                act_size = "scalar"
                self.log_std = jnp.array(log_std_init, dtype=float)

        enc_key, val_key, act_key = jr.split(key, 3)

        self.encoder = MLPNeuralCDE(
            in_size=obs_flat,
            out_size=feature_size,
            latent_size=latent_size,
            solver=solver,
            field_width=field_width,
            field_depth=field_depth,
            initial_width=initial_width,
            initial_depth=initial_depth,
            output_width=output_width,
            output_depth=output_depth,
            time_in_input=False,
            history_length=history_length,
            key=enc_key,
        )

        self.value_model = MLP(
            in_size=feature_size,
            out_size="scalar",
            width_size=value_width,
            depth=value_depth,
            key=val_key,
        )
        self.action_head = MLP(
            in_size=feature_size,
            out_size=act_size,
            width_size=action_width,
            depth=action_depth,
            key=act_key,
        )

    def action_dist_from_features(
        self, features: Float[Array, " feature_size"]
    ) -> AbstractDistribution[ActType]:
        """Return an action distribution from features."""
        action_mean = self.action_head(features)

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

    def _step_encoder(
        self, state: NCDEPolicyState, obs: ObsType
    ) -> tuple[NCDEPolicyState, Float[Array, " feat"]]:
        t_next = state.t + self.dt
        cde_state, y = self.encoder(state.cde, t_next, jnp.ravel(obs))
        return NCDEPolicyState(t=t_next, cde=cde_state), y

    def reset(self) -> NCDEPolicyState:
        return NCDEPolicyState(t=jnp.array(0.0), cde=self.encoder.reset())

    def __call__(
        self, state: NCDEPolicyState, observation: ObsType, *, key: Key | None = None
    ) -> tuple[NCDEPolicyState, ActType]:
        state, features = self._step_encoder(state, observation)
        action_dist = self.action_dist_from_features(features)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return state, action

    def action_and_value(
        self, state: NCDEPolicyState, observation: ObsType, *, key: Key | None = None
    ) -> tuple[NCDEPolicyState, ActType, Float[Array, ""], Float[Array, ""]]:
        state, features = self._step_encoder(state, observation)
        dist = self.action_dist_from_features(features)
        value = self.value_model(features)

        if key is None:
            action = dist.mode()
            log_prob = dist.log_prob(action)
        else:
            action, log_prob = dist.sample_and_log_prob(key)

        return state, action, value, log_prob.sum().squeeze()

    def evaluate_action(
        self, state: NCDEPolicyState, observation: ObsType, action: ActType
    ) -> tuple[NCDEPolicyState, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        features = self.encoder(state.cde, state.t + self.dt, jnp.ravel(observation))[1]
        dist = self.action_dist_from_features(features)
        value = self.value_model(features)
        log_prob = dist.log_prob(action)
        try:
            entropy = dist.entropy().squeeze()
        except NotImplementedError:
            entropy = -log_prob.mean().squeeze()
        return state, value, log_prob.sum().squeeze(), entropy

    def value(
        self, state: NCDEPolicyState, observation: ObsType
    ) -> tuple[NCDEPolicyState, Float[Array, ""]]:
        state, feats = self._step_encoder(state, observation)
        return state, self.value_model(feats)
