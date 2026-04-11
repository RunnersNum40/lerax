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


class A2CStepState[PolicyType: AbstractActorCriticPolicy](AbstractStepState):
    """
    Step-level state for A2C.

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
    ) -> A2CStepState[PolicyType]:
        """Initialize the step state."""
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        callback_states = callback.step_reset(ResetContext(locals()), key=key)
        return cls(env_state, policy_state, callback_states)


class A2CState[PolicyType: AbstractActorCriticPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    Iteration-level state for A2C.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The step-level state.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The optimizer state.
        callback_state: The state of the callback for this iteration.
    """

    iteration_count: Int[Array, ""]
    step_state: A2CStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState


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
    AbstractAlgorithm[PolicyType, A2CState[PolicyType]]
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
        num_steps: int = 5,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        entropy_loss_coefficient: float = 0.0,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = False,
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

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: A2CStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> tuple[A2CStepState[PolicyType], RolloutBuffer]:
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
            A2CStepState(next_env_state, next_policy_state, callback_state),
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
        step_state: A2CStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> tuple[A2CStepState[PolicyType], RolloutBuffer]:
        """Collect a rollout using the current policy."""
        key, post_collect_key = jr.split(key, 2)

        def scan_step(
            carry: A2CStepState[PolicyType], key: Key[Array, ""]
        ) -> tuple[A2CStepState[PolicyType], RolloutBuffer]:
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
    ) -> A2CState[PolicyType]:
        step_key, callback_key = jr.split(key, 2)

        if self.num_envs == 1:
            step_state = A2CStepState.initial(env, policy, callback, step_key)
        else:
            step_state = eqx.filter_vmap(
                A2CStepState.initial, in_axes=(None, None, None, 0)
            )(env, policy, callback, jr.split(step_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        return A2CState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
        )

    def iteration(
        self,
        state: A2CState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> A2CState[PolicyType]:
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

        (_, stats), grads = self.a2c_loss_grad(
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
