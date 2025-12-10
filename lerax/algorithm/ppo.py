from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.callback import AbstractCallback, CallbackList
from lerax.env import AbstractEnvLike
from lerax.policy import AbstractStatefulActorCriticPolicy
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm
from .on_policy import AbstractOnPolicyAlgorithm, OnPolicyState, OnPolicyStepState


class PPOStats(eqx.Module):
    """
    PPO training statistics.

    Attributes:
        approx_kl: Approximate KL divergence between old and new policy.
        total_loss: Total loss.
        policy_loss: Policy loss.
        value_loss: Value function loss.
        entropy_loss: Entropy loss.
    """

    approx_kl: Float[Array, ""]
    total_loss: Float[Array, ""]
    policy_loss: Float[Array, ""]
    value_loss: Float[Array, ""]
    entropy_loss: Float[Array, ""]


LearningRateSpec = float | optax.Schedule | Callable[[int], optax.Schedule]


@eqx.filter_grad(has_aux=True)
def ppo_loss_grad(
    policy: AbstractStatefulActorCriticPolicy,
    rollout_buffer: RolloutBuffer,
    normalize_advantages: bool,
    clip_coefficient: float,
    clip_value_loss: bool,
    value_loss_coefficient: float,
    entropy_loss_coefficient: float,
) -> tuple[Float[Array, ""], PPOStats]:
    _, values, log_probs, entropy = jax.vmap(policy.evaluate_action)(
        rollout_buffer.states,
        rollout_buffer.observations,
        rollout_buffer.actions,
    )

    values = eqx.error_if(values, ~jnp.isfinite(values), "Non-finite values.")
    log_probs = eqx.error_if(
        log_probs,
        ~jnp.isfinite(log_probs),
        "Non-finite log_probs.",
    )
    entropy = eqx.error_if(
        entropy,
        ~jnp.isfinite(entropy),
        "Non-finite entropy.",
    )

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
            * jnp.clip(
                ratios,
                1 - clip_coefficient,
                1 + clip_coefficient,
            ),
        )
    )

    if clip_value_loss:
        clipped_values = rollout_buffer.values + jnp.clip(
            values - rollout_buffer.values,
            -clip_coefficient,
            clip_coefficient,
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
        approx_kl=approx_kl,
        total_loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
    )


class PPORunner[PolicyType: AbstractStatefulActorCriticPolicy](
    AbstractOnPolicyAlgorithm[PolicyType]
):
    """
    Concrete PPO runner.

    This is the fully configured on-policy algorithm that actually performs
    training. It is produced by the PPO builder.
    """

    env: AbstractEnvLike
    callback: AbstractCallback
    optimizer: optax.GradientTransformation

    num_envs: int
    num_steps: int
    total_timesteps: int
    num_iterations: int

    gae_lambda: float
    gamma: float
    batch_size: int

    num_epochs: int
    normalize_advantages: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_loss_coefficient: float
    value_loss_coefficient: float
    max_grad_norm: float

    def per_step(
        self,
        step_state: OnPolicyStepState[PolicyType],
    ) -> OnPolicyStepState[PolicyType]:
        return step_state

    def per_iteration(
        self,
        state: OnPolicyState[PolicyType],
    ) -> OnPolicyState[PolicyType]:
        return state

    def train_batch(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
    ) -> tuple[PolicyType, optax.OptState, PPOStats]:
        grads, stats = ppo_loss_grad(
            policy,
            rollout_buffer,
            self.normalize_advantages,
            self.clip_coefficient,
            self.clip_value_loss,
            self.value_loss_coefficient,
            self.entropy_loss_coefficient,
        )

        updates, new_opt_state = self.optimizer.update(
            grads,
            opt_state,
            eqx.filter(policy, eqx.is_inexact_array),
        )
        policy = eqx.apply_updates(policy, updates)

        return policy, new_opt_state, stats

    def train_epoch(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
        *,
        key: Key,
    ) -> tuple[PolicyType, optax.OptState, PPOStats]:
        def batch_scan(
            carry: tuple[PolicyType, optax.OptState],
            buffer: RolloutBuffer,
        ) -> tuple[tuple[PolicyType, optax.OptState], PPOStats]:
            policy, opt_state = carry
            policy, opt_state, stats = self.train_batch(policy, opt_state, buffer)
            return (policy, opt_state), stats

        (policy, opt_state), stats = filter_scan(
            batch_scan,
            (policy, opt_state),
            rollout_buffer.batches(self.batch_size, key=key),
        )
        stats = jax.tree.map(jnp.mean, stats)
        return policy, opt_state, stats

    @staticmethod
    def explained_variance(
        returns: Float[Array, ""],
        values: Float[Array, ""],
    ) -> Float[Array, ""]:
        variance = jnp.var(returns)
        return 1 - jnp.var(returns - values) / (variance + jnp.finfo(returns.dtype).eps)

    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: RolloutBuffer,
        *,
        key: Key,
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        def epoch_scan(
            carry: tuple[PolicyType, optax.OptState],
            epoch_key: Key,
        ) -> tuple[tuple[PolicyType, optax.OptState], PPOStats]:
            policy, opt_state = carry
            policy, opt_state, stats = self.train_epoch(
                policy,
                opt_state,
                buffer,
                key=epoch_key,
            )
            return (policy, opt_state), stats

        (policy, opt_state), stats = filter_scan(
            epoch_scan,
            (policy, opt_state),
            jr.split(key, self.num_epochs),
        )

        stats = jax.tree.map(jnp.mean, stats)
        explained_variance = self.explained_variance(buffer.returns, buffer.values)
        log: dict[str, Scalar] = {
            "approx_kl": stats.approx_kl,
            "loss": stats.total_loss,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy_loss": stats.entropy_loss,
            "explained_variance": explained_variance,
        }
        return policy, opt_state, log


class PPO[PolicyType: AbstractStatefulActorCriticPolicy](
    AbstractAlgorithm[PPORunner[PolicyType]]
):
    """
    PPO builder.

    Holds hyperparameters, environment binding, horizon, and callback. Produces
    a `PPORunner` via `build()`, and exposes a convenience `learn` method via
    `AbstractAlgorithm`.
    """

    num_envs: int
    env: AbstractEnvLike | None
    total_timesteps: int | None
    callback: AbstractCallback

    num_steps: int
    num_epochs: int
    num_batches: int

    gae_lambda: float
    gamma: float
    clip_coefficient: float
    clip_value_loss: bool
    entropy_loss_coefficient: float
    value_loss_coefficient: float
    max_grad_norm: float
    normalize_advantages: bool

    learning_rate: LearningRateSpec

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
        learning_rate: LearningRateSpec = 3e-4,
        env: AbstractEnvLike | None = None,
        total_timesteps: int | None = None,
        callback: AbstractCallback | Sequence[AbstractCallback] | None = None,
    ):
        self.num_envs = num_envs
        self.env = env
        self.total_timesteps = total_timesteps
        if callback is None:
            self.callback = CallbackList(callbacks=[])
        elif isinstance(callback, AbstractCallback):
            self.callback = callback
        else:
            self.callback = CallbackList(callbacks=list(callback))

        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.num_batches = num_batches

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.clip_coefficient = clip_coefficient
        self.clip_value_loss = clip_value_loss
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages

        self.learning_rate = learning_rate

    def _make_optimizer(
        self,
        total_gradient_steps: int,
    ) -> optax.GradientTransformation:
        lr_spec = self.learning_rate

        if isinstance(lr_spec, (float, int)):
            lr = float(lr_spec)
        elif callable(lr_spec):
            lr = lr_spec(total_gradient_steps)
        else:
            lr = lr_spec

        adam = optax.adam(lr)
        clip = optax.clip_by_global_norm(self.max_grad_norm)
        return optax.chain(clip, adam)

    def build_runner(
        self,
        env: AbstractEnvLike,
        total_timesteps: int,
        callback: AbstractCallback,
    ) -> PPORunner[PolicyType]:
        rollout_size = self.num_envs * self.num_steps
        if rollout_size <= 0:
            raise ValueError(
                "num_envs * num_steps must be positive; "
                f"got num_envs={self.num_envs}, num_steps={self.num_steps}."
            )

        num_iterations = total_timesteps // rollout_size
        if num_iterations <= 0:
            raise ValueError(
                "total_timesteps too small given num_envs and num_steps; "
                f"got total_timesteps={total_timesteps}, rollout_size={rollout_size}."
            )

        batch_size = rollout_size // self.num_batches
        if batch_size <= 0:
            raise ValueError(
                "num_batches too large relative to rollout_size; "
                f"rollout_size={rollout_size}, num_batches={self.num_batches}."
            )

        total_gradient_steps = num_iterations * self.num_epochs * self.num_batches
        optimizer = self._make_optimizer(total_gradient_steps)

        return PPORunner(
            env=env,
            callback=callback,
            optimizer=optimizer,
            num_envs=self.num_envs,
            num_steps=self.num_steps,
            total_timesteps=total_timesteps,
            num_iterations=num_iterations,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            batch_size=batch_size,
            num_epochs=self.num_epochs,
            normalize_advantages=self.normalize_advantages,
            clip_coefficient=self.clip_coefficient,
            clip_value_loss=self.clip_value_loss,
            entropy_loss_coefficient=self.entropy_loss_coefficient,
            value_loss_coefficient=self.value_loss_coefficient,
            max_grad_norm=self.max_grad_norm,
        )
