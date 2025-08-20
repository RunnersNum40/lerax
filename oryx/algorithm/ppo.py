from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, Scalar

from oryx.buffer import RolloutBuffer
from oryx.env import AbstractEnvLike
from oryx.policy import AbstractActorCriticPolicy
from oryx.utils import filter_scan

from .on_policy import AbstractOnPolicyAlgorithm


class PPOStats(eqx.Module):
    approx_kl: Float[Array, ""]
    total_loss: Float[Array, ""]
    policy_loss: Float[Array, ""]
    value_loss: Float[Array, ""]
    entropy_loss: Float[Array, ""]
    state_magnitude_loss: Float[Array, ""]


class PPO[ActType, ObsType](AbstractOnPolicyAlgorithm[ActType, ObsType]):
    """Proximal Policy Optimization (PPO) algorithm."""

    state_index: eqx.nn.StateIndex[optax.OptState]
    env: AbstractEnvLike[ActType, ObsType]
    policy: AbstractActorCriticPolicy[Float, ActType, ObsType]

    gae_lambda: float
    gamma: float
    num_steps: int
    batch_size: int

    optimizer: optax.GradientTransformation
    anneal_learning_rate: bool

    num_epochs: int
    num_mini_batches: int

    normalize_advantages: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_loss_coefficient: float
    value_loss_coefficient: float
    state_magnitude_coefficient: float
    max_grad_norm: float

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        *,
        num_steps: int = 2048,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        anneal_learning_rate: bool = True,
        num_epochs: int = 10,
        num_mini_batches: int = 32,
        clip_coefficient: float = 0.2,
        clip_value_loss: bool = True,
        entropy_loss_coefficient: float = 0.0,
        value_loss_coefficient: float = 0.5,
        state_magnitude_coefficient: float = 0.0,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        optimizer: optax.GradientTransformation | None = None,
    ):
        self.env = env
        self.policy = policy

        self.num_steps = int(num_steps)
        self.gae_lambda = float(gae_lambda)
        self.gamma = float(gamma)

        self.num_epochs = int(num_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.batch_size = self.num_steps // self.num_mini_batches

        self.clip_coefficient = float(clip_coefficient)
        self.clip_value_loss = bool(clip_value_loss)
        self.entropy_loss_coefficient = float(entropy_loss_coefficient)
        self.value_loss_coefficient = float(value_loss_coefficient)
        self.state_magnitude_coefficient = float(state_magnitude_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_advantages = bool(normalize_advantages)

        self.optimizer = self.make_optimizer(
            learning_rate,
            anneal_learning_rate,
            num_epochs,
            num_mini_batches,
            max_grad_norm,
            optimizer,
        )
        self.anneal_learning_rate = bool(anneal_learning_rate)

        opt_state = self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array))
        self.state_index = eqx.nn.StateIndex(opt_state)

    @staticmethod
    def make_optimizer(
        learning_rate: float,
        anneal_learning_rate: bool,
        num_epochs: int,
        num_mini_batches: int,
        max_grad_norm: float,
        optimizer: optax.GradientTransformation | None = None,
    ) -> optax.GradientTransformation:
        if optimizer is None:
            if anneal_learning_rate:
                schedule = optax.linear_schedule(
                    init_value=learning_rate,
                    end_value=0.0,
                    transition_steps=num_epochs * num_mini_batches,
                )
            else:
                schedule = learning_rate

            adam = optax.inject_hyperparams(optax.adam)(
                learning_rate=schedule, eps=1e-5
            )

            optimizer = optax.named_chain(
                ("clipping", optax.clip_by_global_norm(max_grad_norm)),
                ("adam", adam),
            )

        return optimizer

    @staticmethod
    def ppo_loss(
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        normalize_advantages: bool,
        clip_coefficient: float,
        clip_value_loss: bool,
        value_loss_coefficient: float,
        state_magnitude_coefficient: float,
        entropy_loss_coefficient: float,
    ) -> tuple[Float[Array, ""], PPOStats]:
        _, new_values, new_log_probs, entropy = jax.vmap(policy.evaluate_action)(
            rollout_buffer.states, rollout_buffer.observations, rollout_buffer.actions
        )

        log_ratios = new_log_probs - rollout_buffer.log_probs
        ratios = jnp.exp(log_ratios)
        approx_kl = jnp.mean(ratios - log_ratios) - 1

        advantages = rollout_buffer.advantages
        if normalize_advantages:
            advantages = (advantages - jnp.mean(advantages)) / (
                jnp.std(advantages) + jnp.finfo(advantages.dtype).eps
            )

        policy_loss = -jnp.mean(
            jnp.minimum(
                advantages * ratios,
                advantages
                * jnp.clip(ratios, 1 - clip_coefficient, 1 + clip_coefficient),
            )
        )

        if clip_value_loss:
            clipped_values = rollout_buffer.values + jnp.clip(
                new_values - rollout_buffer.values,
                -clip_coefficient,
                clip_coefficient,
            )
            value_loss = (
                jnp.mean(
                    jnp.minimum(
                        jnp.square(new_values - rollout_buffer.returns),
                        jnp.square(clipped_values - rollout_buffer.returns),
                    )
                )
                / 2
            )
        else:
            value_loss = jnp.mean(jnp.square(new_values - rollout_buffer.returns)) / 2

        entropy_loss = jnp.mean(entropy)

        # TODO: Add state magnitude loss
        # manitude loss is proportional to the squared L2 norm of the latent state of the policy
        state_magnitude_loss = jnp.array(0.0)

        loss = (
            policy_loss
            + value_loss * value_loss_coefficient
            + state_magnitude_loss * state_magnitude_coefficient
            - entropy_loss * entropy_loss_coefficient
        )

        return loss, PPOStats(
            approx_kl, loss, policy_loss, value_loss, entropy_loss, state_magnitude_loss
        )

    ppo_loss_grad = staticmethod(eqx.filter_value_and_grad(ppo_loss, has_aux=True))

    def train_batch(
        self,
        state: eqx.nn.State,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        rollout_buffer: RolloutBuffer[ActType, ObsType],
    ) -> tuple[
        eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType], PPOStats
    ]:
        """
        Train the policy for one batch using the rollout buffer.

        Assumes that the rollout buffer is a single batch of data.
        """
        (_, stats), grads = self.ppo_loss_grad(
            policy,
            rollout_buffer,
            self.normalize_advantages,
            self.clip_coefficient,
            self.clip_value_loss,
            self.value_loss_coefficient,
            self.state_magnitude_coefficient,
            self.entropy_loss_coefficient,
        )

        opt_state = state.get(self.state_index)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        policy = eqx.apply_updates(policy, updates)
        state = state.set(self.state_index, opt_state)

        leaves = jax.tree.leaves(grads)
        flat_grads = jnp.concatenate(jax.tree.map(jnp.ravel, leaves))
        nan_in_grads = jnp.any(jnp.isnan(flat_grads))
        stats = eqx.error_if(stats, nan_in_grads, "NaN in gradients")

        return state, policy, stats

    def train_epoch(
        self,
        state: eqx.nn.State,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
    ):
        """
        Train the policy for one epoch using the rollout buffer.

        One epoch consists of multiple mini-batches.
        """

        def batch_scan(
            carry: tuple[
                eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType]
            ],
            rollout_buffer: RolloutBuffer[ActType, ObsType],
        ) -> tuple[
            tuple[eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType]],
            PPOStats,
        ]:
            state, policy = carry
            state, policy, stats = self.train_batch(state, policy, rollout_buffer)
            return (state, policy), stats

        (state, policy), stats = filter_scan(
            batch_scan,
            (state, policy),
            rollout_buffer.batches(self.batch_size, key=key),
        )

        stats = jax.tree.map(jnp.mean, stats)

        return state, policy, stats

    def train(
        self,
        state: eqx.nn.State,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
    ) -> tuple[
        eqx.nn.State,
        AbstractActorCriticPolicy[Float, ActType, ObsType],
        dict[str, Scalar],
    ]:
        def epoch_scan(
            carry: tuple[
                eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType], Key
            ],
            _,
        ) -> tuple[
            tuple[
                eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType], Key
            ],
            PPOStats,
        ]:
            state, policy, key = carry
            epoch_key, carry_key = jr.split(key, 2)
            state, policy, stats = self.train_epoch(
                state, policy, rollout_buffer, key=epoch_key
            )
            return (state, policy, carry_key), stats

        (state, policy, _), stats = filter_scan(
            epoch_scan, (state, policy, key), length=self.num_epochs
        )

        stats = jax.tree.map(jnp.mean, stats)
        variance = jnp.var(rollout_buffer.rewards)
        explained_variance = 1 - jnp.var(
            rollout_buffer.returns - rollout_buffer.values
        ) / (variance + jnp.finfo(rollout_buffer.returns.dtype).eps)
        log = {
            "loss/approx_kl": stats.approx_kl,
            "loss/total": stats.total_loss,
            "loss/policy": stats.policy_loss,
            "loss/value": stats.value_loss,
            "loss/entropy": stats.entropy_loss,
            "loss/state_magnitude": stats.state_magnitude_loss,
            "stats/variance": variance,
            "stats/explained_variance": explained_variance,
        }

        return state, policy, log

    def learning_rate(self, state: eqx.nn.State) -> Float[Array, ""]:
        opt_state = state.get(self.state_index)
        return opt_state["adam"].hyperparams["learning_rate"]  # pyright: ignore

    @classmethod
    def load(cls, path: str) -> PPO[ActType, ObsType]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError
