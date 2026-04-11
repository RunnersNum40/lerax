from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key, Scalar

from lerax.buffer import RolloutBuffer
from lerax.callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    IterationContext,
    ResetContext,
    StepContext,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractActorCriticPolicy, AbstractPolicyState
from lerax.space import Box
from lerax.utils import filter_cond, filter_scan

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class PPOStepState[PolicyType: AbstractActorCriticPolicy](AbstractStepState):
    """
    Step-level state for PPO.

    Attributes:
        env_state: The state of the environment.
        policy_state: The state of the policy.
        callback_state: The state of the callback for this step.
    """

    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    callback_state: AbstractCallbackStepState

    @classmethod
    def initial(
        cls,
        env: AbstractEnvLike,
        policy: PolicyType,
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> PPOStepState[PolicyType]:
        """Initialize the step state."""
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        callback_states = callback.step_reset(ResetContext(locals()), key=key)
        return cls(env_state, policy_state, callback_states)


class PPOState[PolicyType: AbstractActorCriticPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    Iteration-level state for PPO.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The step-level state.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The optimizer state.
        callback_state: The state of the callback for this iteration.
    """

    iteration_count: Int[Array, ""]
    step_state: PPOStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState


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


class PPO[PolicyType: AbstractActorCriticPolicy](
    AbstractAlgorithm[PolicyType, PPOState[PolicyType]]
):
    """
    Proximal Policy Optimization (PPO) algorithm.

    Attributes:
        optimizer: The optimizer used for training.
        gae_lambda: Lambda parameter for Generalized Advantage Estimation (GAE).
        gamma: Discount factor.
        num_envs: Number of parallel environments.
        num_steps: Number of steps to run for each environment per update.
        batch_size: Size of each training batch.
        num_epochs: Number of epochs to train the policy per update.
        normalize_advantages: Whether to normalize advantages.
        clip_coefficient: Clipping coefficient for policy and value function updates.
        clip_value_loss: Whether to clip the value function loss.
        entropy_loss_coefficient: Coefficient for the entropy loss term.
        value_loss_coefficient: Coefficient for the value function loss term.
        max_grad_norm: Maximum gradient norm for gradient clipping.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps to run for each environment per update.
        num_epochs: Number of epochs to train the policy per update.
        num_batches: Number of batches to split the rollout buffer into for training.
        gae_lambda: Lambda parameter for Generalized Advantage Estimation (GAE).
        gamma: Discount factor.
        clip_coefficient: Clipping coefficient for policy and value function updates.
        clip_value_loss: Whether to clip the value function loss.
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
        num_steps: int = 2048,
        num_epochs: int = 10,
        num_batches: int = 32,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        clip_coefficient: float = 0.2,
        clip_value_loss: bool = False,
        entropy_loss_coefficient: float = 0.0,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        learning_rate: optax.ScalarOrSchedule = 3e-4,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.batch_size = (self.num_steps * self.num_envs) // num_batches

        self.clip_coefficient = clip_coefficient
        self.clip_value_loss = clip_value_loss
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages

        adam = optax.inject_hyperparams(optax.adam)(learning_rate)
        clip = optax.clip_by_global_norm(self.max_grad_norm)
        self.optimizer = optax.chain(clip, adam)

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: PPOStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> tuple[PPOStepState[PolicyType], RolloutBuffer]:
        """Perform a single environment step and collect rollout data."""
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            bootstrap_key,
            env_reset_key,
            policy_reset_key,
            callback_key,
        ) = jr.split(key, 9)

        observation = env.observation(state.env_state, key=observation_key)

        action_mask = env.action_mask(state.env_state, key=observation_key)
        next_policy_state, action, value, log_prob = policy.action_and_value(
            state.policy_state, observation, key=action_key, action_mask=action_mask
        )

        if isinstance(env.action_space, Box):
            clipped_action = jnp.clip(
                action,
                env.action_space.low,
                env.action_space.high,
            )
        else:
            clipped_action = action

        next_env_state = env.transition(
            state.env_state, clipped_action, key=transition_key
        )

        reward = env.reward(
            state.env_state, clipped_action, next_env_state, key=reward_key
        )
        termination = env.terminal(next_env_state, key=terminal_key)
        truncation = env.truncate(next_env_state)
        done = termination | truncation

        bootstrapped_reward = lax.cond(
            truncation,
            lambda: (
                reward
                + self.gamma
                * policy.value(
                    next_policy_state,
                    env.observation(next_env_state, key=bootstrap_key),
                )[1]
            ),
            lambda: reward,
        )

        next_env_state = filter_cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )

        next_policy_state = filter_cond(
            done, lambda: policy.reset(key=policy_reset_key), lambda: next_policy_state
        )

        callback_state = callback.on_step(
            StepContext(
                state.callback_state, env, policy, done, bootstrapped_reward, locals()
            ),
            key=callback_key,
        )

        return (
            PPOStepState(next_env_state, next_policy_state, callback_state),
            RolloutBuffer(
                observations=observation,
                actions=clipped_action,
                rewards=bootstrapped_reward,
                dones=done,
                log_probs=log_prob,
                values=value,
                states=state.policy_state,
                action_masks=action_mask,
            ),
        )

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: PPOStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> tuple[PPOStepState[PolicyType], RolloutBuffer]:
        """Collect a rollout using the current policy."""
        key, post_collect_key = jr.split(key, 2)

        def scan_step(
            carry: PPOStepState[PolicyType], key: Key[Array, ""]
        ) -> tuple[PPOStepState[PolicyType], RolloutBuffer]:
            carry, rollout = self.step(env, policy, carry, key=key, callback=callback)
            return carry, rollout

        step_state, rollout_buffer = filter_scan(
            scan_step, step_state, jr.split(key, self.num_steps)
        )

        observation = env.observation(step_state.env_state, key=post_collect_key)
        _, next_value = policy.value(step_state.policy_state, observation)
        rollout_buffer = rollout_buffer.compute_returns_and_advantages(
            next_value, self.gae_lambda, self.gamma
        )
        return step_state, rollout_buffer

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> PPOState[PolicyType]:
        step_key, callback_key = jr.split(key, 2)

        if self.num_envs == 1:
            step_state = PPOStepState.initial(env, policy, callback, step_key)
        else:
            step_state = eqx.filter_vmap(
                PPOStepState.initial, in_axes=(None, None, None, 0)
            )(env, policy, callback, jr.split(step_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        return PPOState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
        )

    def iteration(
        self,
        state: PPOState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> PPOState[PolicyType]:
        rollout_key, train_key, callback_key = jr.split(key, 3)

        if self.num_envs == 1:
            step_state, rollout_buffer = self.collect_rollout(
                state.env, state.policy, state.step_state, callback, rollout_key
            )
        else:
            step_state, rollout_buffer = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, eqx.if_array(0), None, 0)
            )(
                state.env,
                state.policy,
                state.step_state,
                callback,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, log = self.train(
            state.policy, state.opt_state, rollout_buffer, key=train_key
        )

        state = state.next(step_state, policy, opt_state)

        state = state.with_callback_states(
            callback.on_iteration(
                IterationContext(
                    state.callback_state,
                    state.step_state.callback_state,
                    state.env,
                    state.policy,
                    state.iteration_count,
                    state.opt_state,
                    log,
                    self,
                    locals(),
                ),
                key=callback_key,
            )
        )

        state, new_cb = callback.apply_curriculum(state, state.callback_state)
        return state.with_callback_states(new_cb)

    @staticmethod
    def ppo_loss(
        policy: PolicyType,
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
            action_mask=rollout_buffer.action_masks,
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
        policy: PolicyType,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
    ) -> tuple[PolicyType, optax.OptState, PPOStats]:
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
        policy: PolicyType,
        opt_state: optax.OptState,
        rollout_buffer: RolloutBuffer,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, PPOStats]:
        flat_buffer = rollout_buffer.flatten_axes()
        indices = flat_buffer.batch_indices(self.batch_size, key=key)

        def batch_scan(
            carry: tuple[
                PolicyType,
                optax.OptState,
            ],
            batch_indices: Array,
        ):
            policy, opt_state = carry
            batch = flat_buffer.gather(batch_indices)
            policy, opt_state, stats = self.train_batch(policy, opt_state, batch)
            return (policy, opt_state), stats

        (policy, opt_state), stats = filter_scan(
            batch_scan, (policy, opt_state), indices
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
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: RolloutBuffer,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        def epoch_scan(
            carry: tuple[PolicyType, optax.OptState], key: Key[Array, ""]
        ) -> tuple[tuple[PolicyType, optax.OptState], PPOStats]:
            policy, opt_state = carry
            policy, opt_state, stats = self.train_epoch(
                policy, opt_state, buffer, key=key
            )
            return (policy, opt_state), stats

        (policy, opt_state), stats = filter_scan(
            epoch_scan, (policy, opt_state), jr.split(key, self.num_epochs)
        )

        stats = jax.tree.map(jnp.mean, stats)
        explained_variance = self.explained_variance(buffer.returns, buffer.values)
        log = {
            "approx_kl": stats.approx_kl,
            "loss": stats.total_loss,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy_loss": stats.entropy_loss,
            "explained_variance": explained_variance,
        }
        return policy, opt_state, log
