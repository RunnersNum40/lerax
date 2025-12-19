from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
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
from lerax.policy import (
    AbstractActorCriticPolicy,
    AbstractPolicy,
    AbstractPolicyState,
)
from lerax.space import Box
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class OnPolicyStepState[PolicyType: AbstractPolicy](AbstractStepState):
    """
    State for on-policy algorithm steps.

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
        key: Key,
    ) -> OnPolicyStepState[PolicyType]:
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


class OnPolicyState[PolicyType: AbstractPolicy](AbstractAlgorithmState[PolicyType]):
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

    iteration_count: Int[Array, ""]
    step_state: OnPolicyStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState


class AbstractOnPolicyAlgorithm[PolicyType: AbstractActorCriticPolicy](
    AbstractAlgorithm[PolicyType, OnPolicyState[PolicyType]]
):
    """
    Base class for on-policy algorithms.

    Generates rollouts using the current policy and estimates advantages and
    returns using GAE. Trains the policy using the collected rollouts.

    Attributes:
        optimizer: The optimizer used for training the policy.
        gae_lambda: The GAE lambda parameter.
        gamma: The discount factor.
        num_envs: The number of parallel environments.
        num_steps: The number of steps to collect per environment.
        batch_size: The batch size for training.
    """

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    gae_lambda: eqx.AbstractVar[float]
    gamma: eqx.AbstractVar[float]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    @abstractmethod
    def per_step(
        self, step_state: OnPolicyStepState[PolicyType]
    ) -> OnPolicyStepState[PolicyType]:
        """Process the step carry after each step."""

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: OnPolicyStepState[PolicyType],
        *,
        key: Key,
        callback: AbstractCallback,
    ) -> tuple[OnPolicyStepState[PolicyType], RolloutBuffer]:
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
        next_policy_state, action, value, log_prob = policy.action_and_value(
            state.policy_state, observation, key=action_key
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
        # TODO: Check if a non-branched approach is faster
        bootstrapped_reward = lax.cond(
            truncation,
            lambda: reward
            + self.gamma
            * policy.value(
                next_policy_state, env.observation(next_env_state, key=bootstrap_key)
            )[1],
            lambda: reward,
        )

        # Reset environment if done
        next_env_state = lax.cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )

        # Reset policy state if done
        next_policy_state = lax.cond(
            done, lambda: policy.reset(key=policy_reset_key), lambda: next_policy_state
        )

        callback_state = callback.on_step(
            StepContext(
                state.callback_state, env, policy, done, bootstrapped_reward, locals()
            ),
            key=callback_key,
        )

        return (
            OnPolicyStepState(next_env_state, next_policy_state, callback_state),
            RolloutBuffer(
                observations=observation,
                actions=clipped_action,
                rewards=bootstrapped_reward,
                dones=done,
                log_probs=log_prob,
                values=value,
                states=state.policy_state,
            ),
        )

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: OnPolicyStepState[PolicyType],
        callback: AbstractCallback,
        key: Key,
    ) -> tuple[OnPolicyStepState[PolicyType], RolloutBuffer]:
        """Collect a rollout using the current policy."""
        key, observation_key = jr.split(key, 2)

        def scan_step(
            carry: OnPolicyStepState[PolicyType], key: Key
        ) -> tuple[OnPolicyStepState[PolicyType], RolloutBuffer]:
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

        _, next_value = policy.value(
            step_state.policy_state,
            env.observation(step_state.env_state, key=observation_key),
        )
        rollout_buffer = rollout_buffer.compute_returns_and_advantages(
            next_value, self.gae_lambda, self.gamma
        )
        return step_state, rollout_buffer

    @abstractmethod
    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: RolloutBuffer,
        *,
        key: Key,
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
        key: Key,
        callback: AbstractCallback,
    ) -> OnPolicyState[PolicyType]:
        step_key, callback_key = jr.split(key, 2)

        if self.num_envs == 1:
            step_state = OnPolicyStepState.initial(env, policy, callback, step_key)
        else:
            step_state = jax.vmap(
                OnPolicyStepState.initial, in_axes=(None, None, None, 0)
            )(env, policy, callback, jr.split(step_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        return OnPolicyState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
        )

    def iteration(
        self,
        state: OnPolicyState[PolicyType],
        *,
        key: Key,
        callback: AbstractCallback,
    ) -> OnPolicyState[PolicyType]:
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
                self.collect_rollout, in_axes=(None, None, 0, None, 0)
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
                    locals(),
                ),
                key=callback_key,
            )
        )

        return self.per_iteration(state)
