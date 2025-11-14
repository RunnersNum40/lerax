from __future__ import annotations

from typing import ClassVar

import diffrax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, Real

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.model import MLP, MLPNeuralCDE, NCDEState
from lerax.space import AbstractSpace

from .base_actor_critic import (
    AbstractPolicyState,
    AbstractStatefulActorCriticPolicy,
)
from .utils import ActionHead


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

    action_space: AbstractSpace[ActType]
    observation_space: AbstractSpace[ObsType]

    encoder: MLPNeuralCDE
    value_head: MLP
    action_head: ActionHead

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
        value_width: int = 16,
        value_depth: int = 1,
        action_width: int = 16,
        action_depth: int = 1,
        history_length: int = 4,
        dt: float = 1.0,
        log_std_init: float = 0.0,
        key: Key,
    ):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.dt = float(dt)

        enc_key, val_key, act_key = jr.split(key, 3)

        self.encoder = MLPNeuralCDE(
            in_size=int(jnp.array(self.observation_space.flat_size)),
            latent_size=latent_size,
            solver=solver,
            field_width=field_width,
            field_depth=field_depth,
            initial_width=initial_width,
            initial_depth=initial_depth,
            time_in_input=False,
            history_length=history_length,
            key=enc_key,
        )

        self.value_head = MLP(
            in_size=latent_size,
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

    def _step_encoder(
        self, state: NCDEPolicyState, obs: ObsType
    ) -> tuple[NCDEPolicyState, Float[Array, " feat"]]:
        t_next = state.t + self.dt
        cde_state, y = self.encoder(
            state.cde, t_next, self.observation_space.flatten_sample(obs)
        )
        return NCDEPolicyState(t=t_next, cde=cde_state), y

    def reset(self) -> NCDEPolicyState:
        return NCDEPolicyState(t=jnp.array(0.0), cde=self.encoder.reset())

    def __call__(
        self, state: NCDEPolicyState, observation: ObsType, *, key: Key | None = None
    ) -> tuple[NCDEPolicyState, ActType]:
        state, features = self._step_encoder(state, observation)
        action_dist = self.action_head(features)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return state, action

    def action_and_value(
        self, state: NCDEPolicyState, observation: ObsType, *, key: Key
    ) -> tuple[NCDEPolicyState, ActType, Float[Array, ""], Float[Array, ""]]:
        state, features = self._step_encoder(state, observation)

        value = self.value_head(features)

        dist = self.action_head(features)
        action, log_prob = dist.sample_and_log_prob(key)

        return state, action, value, log_prob.sum().squeeze()

    def evaluate_action(
        self, state: NCDEPolicyState, observation: ObsType, action: ActType
    ) -> tuple[NCDEPolicyState, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        state, features = self._step_encoder(state, observation)
        dist = self.action_head(features)
        value = self.value_head(features)
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
        return state, self.value_head(feats)
