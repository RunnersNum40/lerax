from __future__ import annotations

from typing import Any, ClassVar

import diffrax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Integer, Key, Real

from lerax.distribution import AbstractMaskableDistribution
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.model import MLP, ActionLayer, MLPNeuralCDE, NCDEState
from lerax.space import AbstractSpace

from .base_actor_critic import AbstractActorCriticPolicy, AbstractPolicyState


class NCDEPolicyState(AbstractPolicyState):
    t: Float[Array, ""]
    cde: NCDEState

    def __init__(self, *, t: Float[Array, ""] = jnp.array(0.0), cde: NCDEState):
        self.t = jnp.asarray(t, dtype=float)
        self.cde = cde


class NCDEActorCriticPolicy[
    ActType: (Float[Array, " dims"], Integer[Array, ""]),
    ObsType: Real[Array, "..."],
    MaskType,
](AbstractActorCriticPolicy[NCDEPolicyState, ActType, ObsType, MaskType]):
    """
    Actor–critic with a shared MLPNeuralCDE encoder and MLP heads.

    Acts by encoding observations with a Neural CDE, then passing the
    encoded features to separate MLPs to produce action distributions and
    value estimates.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        encoder: Neural CDE to encode observations into features.
        value_head: MLP to produce value estimates from features.
        action_head: MLP to produce action distributions from features.

    Args:
        env: The environment to create the policy for.
        solver: Diffrax solver to use for the Neural CDE.
        feature_size: Size of the feature representation.
        latent_size: Size of the latent state in the Neural CDE.
        field_width: Width of the hidden layers in the Neural CDE vector field.
        field_depth: Depth of the hidden layers in the Neural CDE vector field.
        initial_width: Width of the hidden layers in the Neural CDE initial network.
        initial_depth: Depth of the hidden layers in the Neural CDE initial network.
        value_width: Width of the hidden layers in the value head.
        value_depth: Depth of the hidden layers in the value head.
        action_width: Width of the hidden layers in the action head.
        action_depth: Depth of the hidden layers in the action head.
        history_length: Number of past observations to condition on.
        dt: Time step between observations for the Neural CDE.
        log_std_init: Initial log standard deviation for continuous action spaces.
        key: JAX PRNG key for parameter initialization.
    """

    name: ClassVar[str] = "NCDEActorCriticPolicy"

    action_space: AbstractSpace[ActType, MaskType]
    observation_space: AbstractSpace[ObsType, Any]

    encoder: MLPNeuralCDE
    value_head: MLP
    action_head: ActionLayer

    dt: float = eqx.field(static=True)

    def __init__[StateType: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType, MaskType],
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
            in_size=self.observation_space.flat_size,
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

        self.action_head = ActionLayer(
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

    def reset(self, *, key: Key) -> NCDEPolicyState:
        return NCDEPolicyState(t=jnp.array(0.0), cde=self.encoder.reset())

    def __call__(
        self,
        state: NCDEPolicyState,
        observation: ObsType,
        *,
        key: Key | None = None,
        action_mask: MaskType | None = None,
    ) -> tuple[NCDEPolicyState, ActType]:
        state, features = self._step_encoder(state, observation)
        features = jnp.tanh(features)

        action_dist = self.action_head(features)
        if action_mask is not None and isinstance(
            action_dist, AbstractMaskableDistribution
        ):
            action_dist = action_dist.mask(action_mask)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return state, action

    def action_and_value(
        self,
        state: NCDEPolicyState,
        observation: ObsType,
        *,
        key: Key,
        action_mask: MaskType | None = None,
    ) -> tuple[NCDEPolicyState, ActType, Float[Array, ""], Float[Array, ""]]:
        state, features = self._step_encoder(state, observation)
        features = jnp.tanh(features)

        value = self.value_head(features)

        action_dist = self.action_head(features)
        if action_mask is not None and isinstance(
            action_dist, AbstractMaskableDistribution
        ):
            action_dist = action_dist.mask(action_mask)

        action, log_prob = action_dist.sample_and_log_prob(key)

        return state, action, value, log_prob.sum().squeeze()

    def evaluate_action(
        self,
        state: NCDEPolicyState,
        observation: ObsType,
        action: ActType,
        *,
        action_mask: MaskType | None = None,
    ) -> tuple[NCDEPolicyState, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        state, features = self._step_encoder(state, observation)
        features = jnp.tanh(features)

        action_dist = self.action_head(features)
        if action_mask is not None and isinstance(
            action_dist, AbstractMaskableDistribution
        ):
            action_dist = action_dist.mask(action_mask)

        value = self.value_head(features)
        log_prob = action_dist.log_prob(action)

        eqx.debug.breakpoint_if(~jnp.isfinite(log_prob))

        try:
            entropy = action_dist.entropy().squeeze()
        except NotImplementedError:
            entropy = -log_prob.mean().squeeze()

        return state, value, log_prob.sum().squeeze(), entropy

    def value(
        self, state: NCDEPolicyState, observation: ObsType
    ) -> tuple[NCDEPolicyState, Float[Array, ""]]:
        state, feats = self._step_encoder(state, observation)
        return state, self.value_head(feats)
