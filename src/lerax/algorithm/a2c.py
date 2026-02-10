from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.policy import AbstractActorCriticPolicy

from .on_policy import (
    AbstractActorCriticOnPolicyAlgorithm,
    OnPolicyState,
    OnPolicyStepState,
)


class A2CStats(eqx.Module):
    """
    A2C training statistics.

    Attributes:
        total_loss: Total loss.
        policy_loss: Policy loss.
        value_loss: Value function loss.
        entropy_loss: Entropy loss.
    """

    total_loss: Float[Array, ""]
    policy_loss: Float[Array, ""]
    value_loss: Float[Array, ""]
    entropy_loss: Float[Array, ""]


class A2C[PolicyType: AbstractActorCriticPolicy](
    AbstractActorCriticOnPolicyAlgorithm[PolicyType]
):
    """
    Advantage Actor-Critic (A2C) algorithm.

    Uses GAE for advantage estimation and performs a single gradient update
    per rollout with entropy regularization.

    Attributes:
        optimizer: The optimizer used for training.
        gae_lambda: Lambda parameter for Generalized Advantage Estimation (GAE).
        gamma: Discount factor.
        num_envs: Number of parallel environments.
        num_steps: Number of steps to run for each environment per update.
        batch_size: Size of each training batch.
        normalize_advantages: Whether to normalize advantages.
        entropy_loss_coefficient: Coefficient for the entropy loss term.
        value_loss_coefficient: Coefficient for the value function loss term.
        max_grad_norm: Maximum gradient norm for gradient clipping.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps to run for each environment per update.
        gae_lambda: Lambda parameter for Generalized Advantage Estimation (GAE).
        gamma: Discount factor.
        entropy_loss_coefficient: Coefficient for the entropy loss term.
        value_loss_coefficient: Coefficient for the value function loss term.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        normalize_advantages: Whether to normalize advantages.
        learning_rate: Learning rate for the optimizer.
    """

    optimizer: optax.GradientTransformation

    gae_lambda: float
    gamma: float

    num_envs: int
    num_steps: int
    batch_size: int

    normalize_advantages: bool
    entropy_loss_coefficient: float
    value_loss_coefficient: float
    max_grad_norm: float

    def __init__(
        self,
        *,
        num_envs: int = 4,
        num_steps: int = 512,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        entropy_loss_coefficient: float = 0.01,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        learning_rate: optax.ScalarOrSchedule = 7e-4,
    ):
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = self.num_steps * self.num_envs

        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages

        adam = optax.inject_hyperparams(optax.adam)(learning_rate)
        clip = optax.clip_by_global_norm(self.max_grad_norm)
        self.optimizer = optax.chain(clip, adam)

    def per_step(
        self, step_state: OnPolicyStepState[PolicyType]
    ) -> OnPolicyStepState[PolicyType]:
        return step_state

    def per_iteration(
        self, state: OnPolicyState[PolicyType]
    ) -> OnPolicyState[PolicyType]:
        return state

    @staticmethod
    def a2c_loss(
        policy: PolicyType,
        rollout_buffer: RolloutBuffer,
        normalize_advantages: bool,
        value_loss_coefficient: float,
        entropy_loss_coefficient: float,
    ) -> tuple[Float[Array, ""], A2CStats]:
        _, values, log_probs, entropy = jax.vmap(policy.evaluate_action)(
            rollout_buffer.states,
            rollout_buffer.observations,
            rollout_buffer.actions,
            action_mask=rollout_buffer.action_masks,
        )

        advantages = rollout_buffer.advantages
        if normalize_advantages:
            advantages = (advantages - jnp.mean(advantages)) / (
                jnp.std(advantages) + jnp.finfo(advantages.dtype).eps
            )

        policy_loss = -jnp.mean(log_probs * advantages)
        value_loss = jnp.mean(jnp.square(values - rollout_buffer.returns)) / 2
        entropy_loss = -jnp.mean(entropy)

        loss = (
            policy_loss
            + value_loss * value_loss_coefficient
            + entropy_loss * entropy_loss_coefficient
        )

        return loss, A2CStats(loss, policy_loss, value_loss, entropy_loss)

    a2c_loss_grad = staticmethod(eqx.filter_value_and_grad(a2c_loss, has_aux=True))

    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: RolloutBuffer,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        flat_buffer = buffer.flatten_axes()

        (_, stats), grads = self.a2c_loss_grad(  # type: ignore[missing-argument]
            policy,
            flat_buffer,
            self.normalize_advantages,
            self.value_loss_coefficient,
            self.entropy_loss_coefficient,
        )

        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, updates)

        log = {
            "loss": stats.total_loss,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy_loss": stats.entropy_loss,
        }
        return policy, opt_state, log
