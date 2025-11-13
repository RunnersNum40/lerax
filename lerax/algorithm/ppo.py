from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.policy import AbstractStatefulActorCriticPolicy
from lerax.utils import filter_scan

from .on_policy import AbstractOnPolicyAlgorithm


class PPOStats(eqx.Module):
    approx_kl: Float[Array, ""]
    total_loss: Float[Array, ""]
    policy_loss: Float[Array, ""]
    value_loss: Float[Array, ""]
    entropy_loss: Float[Array, ""]


class PPO(AbstractOnPolicyAlgorithm):
    optimizer: optax.GradientTransformation

    gae_lambda: float
    gamma: float

    num_envs: int
    num_steps: int
    batch_size: int
    num_epochs: int

    normalize_advantages: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_loss_coefficient: float
    value_loss_coefficient: float
    max_grad_norm: float

    def __init__(
        self,
        *,
        num_envs: int = 4,
        num_steps: int = 512,
        num_epochs: int = 16,
        num_batches: int = 32,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        clip_coefficient: float = 0.2,
        clip_value_loss: bool = False,
        entropy_loss_coefficient: float = 0.0,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        learning_rate: float = 3e-4,
    ):
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.batch_size = self.num_steps // num_batches

        self.clip_coefficient = clip_coefficient
        self.clip_value_loss = clip_value_loss
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages

        adam = optax.inject_hyperparams(optax.adam)(learning_rate)
        clip = optax.clip_by_global_norm(self.max_grad_norm)
        self.optimizer = optax.chain(clip, adam)

    # Needs to be static so the first argument can be a policy
    # eqx.filter_value_and_grad doesn't support argnums
    @staticmethod
    def ppo_loss(
        policy: AbstractStatefulActorCriticPolicy,
        rollout_buffer: RolloutBuffer,
        normalize_advantages: bool,
        clip_coefficient: float,
        clip_value_loss: bool,
        value_loss_coefficient: float,
        entropy_loss_coefficient: float,
    ) -> tuple[Float[Array, ""], PPOStats]:
        _, values, log_probs, entropy = jax.vmap(policy.evaluate_action)(
            rollout_buffer.states, rollout_buffer.observations, rollout_buffer.actions
        )

        values = eqx.error_if(values, ~jnp.isfinite(values), "Non-finite values.")
        log_probs = eqx.error_if(
            log_probs, ~jnp.isfinite(log_probs), "Non-finite log_probs."
        )
        entropy = eqx.error_if(entropy, ~jnp.isfinite(entropy), "Non-finite entropy.")

        log_ratios = log_probs - rollout_buffer.log_probs
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
                values - rollout_buffer.values, -clip_coefficient, clip_coefficient
            )
            value_loss = (
                jnp.mean(
                    jnp.minimum(
                        jnp.square(values - rollout_buffer.returns),
                        jnp.square(clipped_values - rollout_buffer.returns),
                    )
                )
                / 2
            )
        else:
            value_loss = jnp.mean(jnp.square(values - rollout_buffer.returns)) / 2

        entropy_loss = -jnp.mean(entropy)

        loss = (
            policy_loss
            + value_loss * value_loss_coefficient
            + entropy_loss * entropy_loss_coefficient
        )

        return loss, PPOStats(
            approx_kl,
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
        )

    ppo_loss_grad = staticmethod(eqx.filter_value_and_grad(ppo_loss, has_aux=True))

    def train_batch(
        self,
        policy: AbstractStatefulActorCriticPolicy,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
    ) -> tuple[AbstractStatefulActorCriticPolicy, optax.OptState, PPOStats]:
        (_, stats), grads = self.ppo_loss_grad(
            policy,
            rollout_buffer,
            self.normalize_advantages,
            self.clip_coefficient,
            self.clip_value_loss,
            self.value_loss_coefficient,
            self.entropy_loss_coefficient,
        )

        updates, new_opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, updates)

        return policy, new_opt_state, stats

    def train_epoch(
        self,
        policy: AbstractStatefulActorCriticPolicy,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
        *,
        key: Key,
    ) -> tuple[AbstractStatefulActorCriticPolicy, optax.OptState, PPOStats]:
        def batch_scan(
            carry: tuple[
                AbstractStatefulActorCriticPolicy,
                optax.OptState,
            ],
            rb: RolloutBuffer,
        ):
            pol, opt_st = carry
            pol, opt_st, stats = self.train_batch(pol, opt_st, rb)
            return (pol, opt_st), stats

        (policy, opt_state), stats = filter_scan(
            batch_scan,
            (policy, opt_state),
            rollout_buffer.batches(self.batch_size, key=key),
        )
        stats = jax.tree.map(jnp.mean, stats)
        return policy, opt_state, stats

    @staticmethod
    def explained_variance(
        returns: Float[Array, ""], values: Float[Array, ""]
    ) -> Float[Array, ""]:
        variance = jnp.var(returns)
        return 1 - jnp.var(returns - values) / (variance + jnp.finfo(returns.dtype).eps)

    def train(
        self,
        policy: AbstractStatefulActorCriticPolicy,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
        *,
        key: Key,
    ) -> tuple[AbstractStatefulActorCriticPolicy, optax.OptState, dict[str, Scalar]]:
        def epoch_scan(
            carry: tuple[AbstractStatefulActorCriticPolicy, optax.OptState, Key], _
        ):
            pol, opt_st, ky = carry
            epoch_key, next_key = jr.split(ky, 2)
            pol, opt_st, stats = self.train_epoch(
                pol, opt_st, rollout_buffer, key=epoch_key
            )
            return (pol, opt_st, next_key), stats

        (policy, opt_state, _), stats = filter_scan(
            epoch_scan, (policy, opt_state, key), length=self.num_epochs
        )

        stats = jax.tree.map(jnp.mean, stats)
        explained_variance = self.explained_variance(
            rollout_buffer.returns, rollout_buffer.values
        )
        log = {
            "approx_kl": stats.approx_kl,
            "loss": stats.total_loss,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy_loss": stats.entropy_loss,
            "explained_variance": explained_variance,
        }
        return policy, opt_state, log
