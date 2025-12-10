from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Int, Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    IterationContext,
    ResetContext,
    StepContext,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicyState, AbstractStatefulPolicy
from lerax.space import Box
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class OffPolicyStepState[PolicyType: AbstractStatefulPolicy](AbstractStepState):
    """
    State object for off-policy algorithms steps.

    Attributes:
        env_state: The state of the environment.
        policy_state: The state of the policy.
        callback_state: The state of the callback for this step.
        buffer: The replay buffer storing experience.
    """

    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    callback_state: AbstractCallbackStepState
    buffer: ReplayBuffer

    @classmethod
    def initial(
        cls,
        size: int,
        env: AbstractEnvLike,
        policy: PolicyType,
        callback: AbstractCallback,
        key: Key,
    ) -> OffPolicyStepState[PolicyType]:
        """
        Initialize the off-policy step state.

        Resets the environment and policy, initializes the callback state,
        and creates an empty replay buffer.

        Args:
            size: The size of the replay buffer.
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

        callback_state = callback.step_reset(ResetContext(locals()), key=key)

        buffer = ReplayBuffer(
            size, env.observation_space, env.action_space, policy_state
        )
        return cls(env_state, policy_state, callback_state, buffer)


class OffPolicyState[PolicyType: AbstractStatefulPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    State for off-policy algorithms.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The current step state.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The optimizer state.
        callback_state: The state of the callback for this iteration.
    """

    iteration_count: Int[Array, ""]
    step_state: OffPolicyStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState


class AbstractOffPolicyAlgorithm[PolicyType: AbstractStatefulPolicy](
    AbstractAlgorithm[PolicyType, OffPolicyState[PolicyType]]
):
    """
    Base class for off-policy algorithms.

    Generates experience using a policy and environment, stores it in a replay buffer,
    and trains the policy using samples from the replay buffer.

    Attributes:
        optimizer: The optimizer used for training the policy.
        buffer_size: The size of the replay buffer.
        gamma: The discount factor for future rewards.
        learning_starts: The number of initial steps to collect before training.
        num_envs: The number of parallel environments.
        batch_size: The batch size for training.
    """

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    buffer_size: eqx.AbstractVar[int]
    gamma: eqx.AbstractVar[float]
    learning_starts: eqx.AbstractVar[int]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    @abstractmethod
    def per_step(
        self, step_state: OffPolicyStepState[PolicyType]
    ) -> OffPolicyStepState[PolicyType]:
        """Process the step carry after each step."""

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: OffPolicyStepState[PolicyType],
        *,
        key: Key,
        callback: AbstractCallback,
    ) -> OffPolicyStepState[PolicyType]:
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            next_observation_key,
            env_reset_key,
            policy_reset_key,
            callback_key,
        ) = jr.split(key, 9)

        observation = env.observation(state.env_state, key=observation_key)
        policy_state, action = policy(state.policy_state, observation, key=action_key)

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

        reward = env.reward(state.env_state, action, next_env_state, key=reward_key)
        termination = env.terminal(next_env_state, key=terminal_key)
        truncation = env.truncate(next_env_state)
        done = termination | truncation
        timeout = truncation & ~termination
        next_observation = env.observation(next_env_state, key=next_observation_key)

        next_env_state = lax.cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )

        next_policy_state = lax.cond(
            done, lambda: policy.reset(key=policy_reset_key), lambda: policy_state
        )

        replay_buffer = state.buffer.add(
            observation,
            next_observation,
            action,
            reward,
            done,
            timeout,
            state.policy_state,
            policy_state,
        )

        callback_state = callback.on_step(
            StepContext(state.callback_state, env, policy, done, reward, locals()),
            key=callback_key,
        )

        return OffPolicyStepState(
            next_env_state, next_policy_state, callback_state, replay_buffer
        )

    def collect_learning_starts(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: OffPolicyStepState[PolicyType],
        callback: AbstractCallback,
        key: Key,
    ) -> OffPolicyStepState[PolicyType]:
        def scan_step(
            carry: OffPolicyStepState, key: Key
        ) -> tuple[OffPolicyStepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return carry, None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.learning_starts)
        )

        return step_state

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: OffPolicyStepState[PolicyType],
        callback: AbstractCallback,
        key: Key,
    ) -> OffPolicyStepState[PolicyType]:
        def scan_step(
            carry: OffPolicyStepState, key: Key
        ) -> tuple[OffPolicyStepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return self.per_step(carry), None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.num_steps)
        )

        return step_state

    @abstractmethod
    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        *,
        key: Key,
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        """
        Trains the policy using data from the replay buffer.

        Args:
            policy: The policy to train.
            opt_state: The current optimizer state.
            buffer: The replay buffer containing experience.
            key: A JAX PRNG key.

        Returns:
            A tuple containing the updated policy, updated optimizer state,
            and a log dictionary with training information.
        """

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key,
        callback: AbstractCallback,
    ) -> OffPolicyState[PolicyType]:
        init_key, starts_key, callback_key = jr.split(key, 3)
        if self.num_envs == 1:
            step_state = OffPolicyStepState.initial(
                self.buffer_size, env, policy, callback, init_key
            )
            step_state = self.collect_learning_starts(
                env, policy, step_state, callback, starts_key
            )
        else:
            step_state = jax.vmap(
                OffPolicyStepState.initial, in_axes=(None, None, None, None, 0)
            )(
                self.buffer_size // self.num_envs,
                env,
                policy,
                callback,
                jr.split(init_key, self.num_envs),
            )
            step_state = jax.vmap(
                self.collect_learning_starts, in_axes=(None, None, 0, None, 0)
            )(env, policy, step_state, callback, jr.split(starts_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        return OffPolicyState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
        )

    def iteration(
        self,
        state: OffPolicyState[PolicyType],
        *,
        key: Key,
        callback: AbstractCallback,
    ) -> OffPolicyState[PolicyType]:
        rollout_key, train_key, callback_key = jr.split(key, 3)

        if self.num_envs == 1:
            step_state = self.collect_rollout(
                state.env, state.policy, state.step_state, callback, rollout_key
            )
        else:
            step_state = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, 0, None, 0)
            )(
                state.env,
                state.policy,
                state.step_state,
                callback,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, log = self.train(
            state.policy, state.opt_state, step_state.buffer, key=train_key
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
