from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Int, Key, Scalar

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
from lerax.policy import AbstractActorCriticPolicy, AbstractPolicy, AbstractPolicyState
from lerax.space import Box
from lerax.utils import filter_cond, filter_scan

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class AbstractOnPolicyStepState[PolicyType: AbstractPolicy](AbstractStepState):
    """
    State for on-policy algorithm steps.

    Attributes:
        env_state: The state of the environment.
        policy_state: The state of the policy.
        callback_state: The state of the callback for this step.
    """

    env_state: eqx.AbstractVar[AbstractEnvLikeState]
    policy_state: eqx.AbstractVar[AbstractPolicyState]
    callback_state: eqx.AbstractVar[AbstractCallbackStepState]

    @classmethod
    def initial(
        cls,
        env: AbstractEnvLike,
        policy: PolicyType,
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> AbstractOnPolicyStepState[PolicyType]:
        """
        Initialize the step state for the on-policy algorithm.

        Resets the environment, policy, and callback states.

        Args:
            env: The environment to initialize.
            policy: The policy to initialize.
            callback: The callback to initialize.
            key: A JAX PRNG key.

        Returns:
            The initialized step state.
        """
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)

        callback_states = callback.step_reset(ResetContext(locals()), key=key)

        return cls(env_state, policy_state, callback_states)


class AbstractOnPolicyState[PolicyType: AbstractPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    State for on-policy algorithms.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The state for the current step.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The optimizer state.
        callback_state: The state of the callback for this iteration.
    """

    iteration_count: eqx.AbstractVar[Int[Array, ""]]
    step_state: eqx.AbstractVar[AbstractOnPolicyStepState[PolicyType]]
    env: eqx.AbstractVar[AbstractEnvLike]
    policy: eqx.AbstractVar[PolicyType]
    opt_state: eqx.AbstractVar[optax.OptState]
    callback_state: eqx.AbstractVar[AbstractCallbackState]


class AbstractOnPolicyAlgorithm[PolicyType: AbstractPolicy](
    AbstractAlgorithm[PolicyType, AbstractOnPolicyState[PolicyType]]
):
    """
    Base class for on-policy algorithms.

    Collects rollouts using the current policy and trains the policy
    using the collected data. Subclasses implement ``step`` to define
    how actions are selected and what data is collected, ``post_collect``
    to define any post-collection processing, and ``train`` to define
    the training procedure.

    Attributes:
        optimizer: The optimizer used for training the policy.
        gamma: The discount factor.
        num_envs: The number of parallel environments.
        num_steps: The number of steps to collect per environment.
        batch_size: The batch size for training.
    """

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    gamma: eqx.AbstractVar[float]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    @abstractmethod
    def per_step(
        self, step_state: AbstractOnPolicyStepState[PolicyType]
    ) -> AbstractOnPolicyStepState[PolicyType]:
        """Process the step carry after each step."""

    @abstractmethod
    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: AbstractOnPolicyStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> tuple[AbstractOnPolicyStepState[PolicyType], RolloutBuffer]:
        """
        Perform a single environment step and collect rollout data.

        Args:
            env: The environment.
            policy: The current policy.
            state: The current step state.
            key: A JAX PRNG key.
            callback: The callback for this step.

        Returns:
            A tuple of the new step state and a rollout buffer entry.
        """

    def post_collect(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: AbstractOnPolicyStepState[PolicyType],
        buffer: RolloutBuffer,
        *,
        key: Key[Array, ""],
    ) -> RolloutBuffer:
        """
        Process the rollout buffer after collection.

        Override to compute returns, advantages, or other post-collection
        processing. By default returns the buffer unchanged.

        Args:
            env: The environment.
            policy: The current policy.
            step_state: The step state after the rollout.
            buffer: The collected rollout buffer.
            key: A JAX PRNG key.

        Returns:
            The processed rollout buffer.
        """
        return buffer

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: AbstractOnPolicyStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> tuple[AbstractOnPolicyStepState[PolicyType], RolloutBuffer]:
        """Collect a rollout using the current policy."""
        key, post_collect_key = jr.split(key, 2)

        def scan_step(
            carry: AbstractOnPolicyStepState[PolicyType], key: Key[Array, ""]
        ) -> tuple[AbstractOnPolicyStepState[PolicyType], RolloutBuffer]:
            carry, rollout = self.step(
                env,
                policy,
                carry,
                key=key,
                callback=callback,
            )
            return self.per_step(carry), rollout

        step_state, rollout_buffer = filter_scan(
            scan_step, step_state, jr.split(key, self.num_steps)
        )

        rollout_buffer = self.post_collect(
            env, policy, step_state, rollout_buffer, key=post_collect_key
        )
        return step_state, rollout_buffer

    @abstractmethod
    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: RolloutBuffer,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        """
        Train the policy using the rollout buffer.

        Args:
            policy: The current policy.
            opt_state: The current optimizer state.
            buffer: The rollout buffer containing collected experiences.
            key: A JAX PRNG key.

        Returns:
            A tuple containing the updated policy, updated optimizer state,
            and a log dictionary.
        """

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> AbstractOnPolicyState[PolicyType]:
        step_key, callback_key = jr.split(key, 2)

        if self.num_envs == 1:
            step_state = AbstractOnPolicyStepState.initial(
                env, policy, callback, step_key
            )
        else:
            step_state = eqx.filter_vmap(
                AbstractOnPolicyStepState.initial, in_axes=(None, None, None, 0)
            )(env, policy, callback, jr.split(step_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        return AbstractOnPolicyState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
        )

    def iteration(
        self,
        state: AbstractOnPolicyState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> AbstractOnPolicyState[PolicyType]:
        rollout_key, train_key, callback_key = jr.split(key, 3)

        if self.num_envs == 1:
            step_state, rollout_buffer = self.collect_rollout(
                state.env,
                state.policy,
                state.step_state,
                callback,
                rollout_key,
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

        return self.per_iteration(state)


class AbstractActorCriticOnPolicyAlgorithm[PolicyType: AbstractActorCriticPolicy](
    AbstractOnPolicyAlgorithm[PolicyType]
):
    """
    Base class for on-policy algorithms that use actor-critic policies.

    Provides a concrete ``step`` that uses the actor-critic interface
    (``action_and_value``, ``value``) and computes GAE advantages after
    rollout collection.

    Attributes:
        gae_lambda: The GAE lambda parameter.
    """

    gae_lambda: eqx.AbstractVar[float]

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: AbstractOnPolicyStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> tuple[AbstractOnPolicyStepState[PolicyType], RolloutBuffer]:
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

        # Bootstrap reward if truncated
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

        # Reset environment if done
        next_env_state = filter_cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )

        # Reset policy state if done
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
            AbstractOnPolicyStepState(
                next_env_state, next_policy_state, callback_state
            ),
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

    def post_collect(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: AbstractOnPolicyStepState[PolicyType],
        buffer: RolloutBuffer,
        *,
        key: Key[Array, ""],
    ) -> RolloutBuffer:
        """Compute returns and advantages using GAE after rollout collection."""
        observation = env.observation(step_state.env_state, key=key)
        _, next_value = policy.value(step_state.policy_state, observation)
        return buffer.compute_returns_and_advantages(
            next_value, self.gae_lambda, self.gamma
        )
