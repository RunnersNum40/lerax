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
    AbstractIterationCallback,
    AbstractStepCallback,
    AbstractVectorizedCallback,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicyState, AbstractStatefulPolicy
from lerax.utils import filter_scan

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class OffPolicyStepState[PolicyType: AbstractStatefulPolicy](AbstractStepState):
    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    callback_states: list[AbstractCallbackStepState | None]
    buffer: ReplayBuffer

    @classmethod
    def initial(
        cls,
        size: int,
        env: AbstractEnvLike,
        policy: PolicyType,
        callbacks: list[AbstractCallback],
        key: Key,
    ) -> OffPolicyStepState[PolicyType]:
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)

        callback_states = [
            (
                callback.step_reset(locals(), key=key)
                if isinstance(callback, AbstractVectorizedCallback)
                else None
            )
            for callback in callbacks
        ]

        buffer = ReplayBuffer(
            size, env.observation_space, env.action_space, policy_state
        )
        return cls(env_state, policy_state, callback_states, buffer)


class OffPolicyState[PolicyType: AbstractStatefulPolicy](
    AbstractAlgorithmState[PolicyType]
):
    iteration_count: Int[Array, ""]
    step_state: OffPolicyStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_states: list[AbstractCallbackState]


class AbstractOffPolicyAlgorithm[PolicyType: AbstractStatefulPolicy](
    AbstractAlgorithm[PolicyType, OffPolicyState[PolicyType]]
):
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
        callbacks: list[AbstractCallback],
    ) -> OffPolicyStepState[PolicyType]:
        (
            start_callback_key,
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            next_observation_key,
            env_reset_key,
            policy_reset_key,
            end_callback_key,
        ) = jr.split(key, 10)

        callback_states = state.callback_states
        callback_states = [
            (
                callback.on_step_start(state, locals(), key=start_callback_key)
                if isinstance(callback, AbstractStepCallback)
                else state
            )
            for callback, state in zip(callbacks, callback_states)
        ]

        observation = env.observation(state.env_state, key=observation_key)
        policy_state, action = policy(state.policy_state, observation, key=action_key)

        next_env_state = env.transition(state.env_state, action, key=transition_key)

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

        callback_states = [
            (
                callback.on_step_end(state, locals(), key=end_callback_key)
                if isinstance(callback, AbstractStepCallback)
                else state
            )
            for callback, state in zip(callbacks, callback_states)
        ]

        return OffPolicyStepState(
            next_env_state, next_policy_state, callback_states, replay_buffer
        )

    def collect_learning_starts(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: OffPolicyStepState[PolicyType],
        callbacks: list[AbstractCallback],
        key: Key,
    ) -> OffPolicyStepState[PolicyType]:
        def scan_step(
            carry: OffPolicyStepState, key: Key
        ) -> tuple[OffPolicyStepState, None]:
            carry = self.step(env, policy, carry, key=key, callbacks=callbacks)
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
        callbacks: list[AbstractCallback],
        key: Key,
    ) -> OffPolicyStepState[PolicyType]:
        def scan_step(
            carry: OffPolicyStepState, key: Key
        ) -> tuple[OffPolicyStepState, None]:
            carry = self.step(env, policy, carry, key=key, callbacks=callbacks)
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
        """Trains the policy using data from the replay buffer."""

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key,
        callbacks: list[AbstractCallback],
    ) -> OffPolicyState[PolicyType]:
        init_key, starts_key, callback_key = jr.split(key, 3)
        if self.num_envs == 1:
            step_state = OffPolicyStepState.initial(
                self.buffer_size, env, policy, callbacks, init_key
            )
            step_state = self.collect_learning_starts(
                env, policy, step_state, callbacks, starts_key
            )
        else:
            step_state = jax.vmap(
                OffPolicyStepState.initial, in_axes=(None, None, None, None, 0)
            )(
                self.buffer_size // self.num_envs,
                env,
                policy,
                callbacks,
                jr.split(init_key, self.num_envs),
            )
            step_state = jax.vmap(
                self.collect_learning_starts, in_axes=(None, None, 0, None, 0)
            )(env, policy, step_state, callbacks, jr.split(starts_key, self.num_envs))

        callback_states = [
            callback.reset(locals(), key=callback_key) for callback in callbacks
        ]

        return OffPolicyState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_states,
        )

    def iteration(
        self,
        state: OffPolicyState[PolicyType],
        *,
        key: Key,
        callbacks: list[AbstractCallback],
    ) -> OffPolicyState[PolicyType]:
        callback_start_key, rollout_key, train_key, callback_end_key = jr.split(key, 4)

        callback_states = state.callback_states
        callback_states = [
            (
                callback.on_iteration_start(state, locals(), key=callback_start_key)
                if isinstance(callback, AbstractIterationCallback)
                else state
            )
            for callback, state in zip(callbacks, callback_states)
        ]

        if self.num_envs == 1:
            step_state = self.collect_rollout(
                state.env, state.policy, state.step_state, callbacks, rollout_key
            )
        else:
            step_state = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, 0, None, 0)
            )(
                state.env,
                state.policy,
                state.step_state,
                callbacks,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, log = self.train(
            state.policy, state.opt_state, step_state.buffer, key=train_key
        )

        state = state.next(step_state, policy, opt_state)

        callback_states = [
            (
                callback.on_iteration_end(
                    callback_state, locals(), key=callback_end_key
                )
                if isinstance(callback, AbstractIterationCallback)
                else callback_state
            )
            for callback, callback_state in zip(callbacks, callback_states)
        ]

        return self.per_iteration(state)
