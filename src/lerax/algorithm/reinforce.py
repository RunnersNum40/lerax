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


class REINFORCEStepState[PolicyType: AbstractActorCriticPolicy](AbstractStepState):
    """
    Step-level state for REINFORCE.

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
    ) -> REINFORCEStepState[PolicyType]:
        """Initialize the step state."""
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        callback_states = callback.step_reset(ResetContext(locals()), key=key)
        return cls(env_state, policy_state, callback_states)


class REINFORCEState[PolicyType: AbstractActorCriticPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    Iteration-level state for REINFORCE.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The step-level state.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The optimizer state.
        callback_state: The state of the callback for this iteration.
    """

    iteration_count: Int[Array, ""]
    step_state: REINFORCEStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState


class REINFORCEStats(eqx.Module):
    """
    REINFORCE training statistics.

    Attributes:
        total_loss: Total loss.
        policy_loss: Policy loss.
        value_loss: Value function loss.
    """

    total_loss: Float[Array, ""]
    policy_loss: Float[Array, ""]
    value_loss: Float[Array, ""]


class REINFORCE[PolicyType: AbstractActorCriticPolicy](
    AbstractAlgorithm[PolicyType, REINFORCEState[PolicyType]]
):
    """
    REINFORCE algorithm with value function baseline.

    Uses Monte Carlo returns (GAE with lambda=1) and vanilla policy
    gradient without clipping or importance sampling.

    Attributes:
        optimizer: The optimizer used for training.
        gae_lambda: Lambda parameter for GAE, fixed to 1.0 for Monte Carlo returns.
        gamma: Discount factor.
        num_envs: Number of parallel environments.
        num_steps: Number of steps to run for each environment per update.
        batch_size: Size of each training batch.
        normalize_advantages: Whether to normalize advantages.
        value_loss_coefficient: Coefficient for the value function loss term.
        max_grad_norm: Maximum gradient norm for gradient clipping.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps to run for each environment per update.
        gamma: Discount factor.
        normalize_advantages: Whether to normalize advantages.
        value_loss_coefficient: Coefficient for the value function loss term.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        learning_rate: Learning rate for the optimizer.
    """

    optimizer: optax.GradientTransformation

    gae_lambda: float
    gamma: float

    num_envs: int
    num_steps: int
    batch_size: int

    normalize_advantages: bool
    value_loss_coefficient: float
    max_grad_norm: float

    def __init__(
        self,
        *,
        num_envs: int = 1,
        num_steps: int = 512,
        gamma: float = 0.99,
        normalize_advantages: bool = True,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_rate: optax.ScalarOrSchedule = 3e-4,
    ):
        self.gae_lambda = 1.0
        self.gamma = gamma

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = self.num_steps * self.num_envs

        self.normalize_advantages = normalize_advantages
        self.value_loss_coefficient = value_loss_coefficient
        self.max_grad_norm = max_grad_norm

        adam = optax.inject_hyperparams(optax.adam)(learning_rate)
        clip = optax.clip_by_global_norm(self.max_grad_norm)
        self.optimizer = optax.chain(clip, adam)

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    # ── Step & rollout collection ──────────────────────────────────────

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: REINFORCEStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> tuple[REINFORCEStepState[PolicyType], RolloutBuffer]:
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
            REINFORCEStepState(next_env_state, next_policy_state, callback_state),
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
        step_state: REINFORCEStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> tuple[REINFORCEStepState[PolicyType], RolloutBuffer]:
        """Collect a rollout using the current policy."""
        key, post_collect_key = jr.split(key, 2)

        def scan_step(
            carry: REINFORCEStepState[PolicyType], key: Key[Array, ""]
        ) -> tuple[REINFORCEStepState[PolicyType], RolloutBuffer]:
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

    # ── Reset & iteration ──────────────────────────────────────────────

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> REINFORCEState[PolicyType]:
        step_key, callback_key = jr.split(key, 2)

        if self.num_envs == 1:
            step_state = REINFORCEStepState.initial(env, policy, callback, step_key)
        else:
            step_state = eqx.filter_vmap(
                REINFORCEStepState.initial, in_axes=(None, None, None, 0)
            )(env, policy, callback, jr.split(step_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        return REINFORCEState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
        )

    def iteration(
        self,
        state: REINFORCEState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> REINFORCEState[PolicyType]:
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

        return state

    # ── Training ───────────────────────────────────────────────────────

    @staticmethod
    def reinforce_loss(
        policy: PolicyType,
        rollout_buffer: RolloutBuffer,
        normalize_advantages: bool,
        value_loss_coefficient: float,
    ) -> tuple[Float[Array, ""], REINFORCEStats]:
        _, values, log_probs, _ = jax.vmap(policy.evaluate_action)(
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

        loss = policy_loss + value_loss * value_loss_coefficient

        return loss, REINFORCEStats(loss, policy_loss, value_loss)

    reinforce_loss_grad = staticmethod(
        eqx.filter_value_and_grad(reinforce_loss, has_aux=True)
    )

    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: RolloutBuffer,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        flat_buffer = buffer.flatten_axes()

        (_, stats), grads = self.reinforce_loss_grad(
            policy,
            flat_buffer,
            self.normalize_advantages,
            self.value_loss_coefficient,
        )

        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, updates)

        log = {
            "loss": stats.total_loss,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
        }
        return policy, opt_state, log
