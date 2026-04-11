from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key, Scalar

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
from lerax.policy import AbstractPolicyState
from lerax.policy.q import AbstractQPolicy
from lerax.space import Box
from lerax.utils import filter_cond, filter_scan

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class DQNStepState[PolicyType: AbstractQPolicy](AbstractStepState):
    """
    Step-level state for DQN.

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
        key: Key[Array, ""],
    ) -> DQNStepState[PolicyType]:
        """Initialize the step state with an empty replay buffer."""
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        callback_state = callback.step_reset(ResetContext(locals()), key=key)
        buffer = ReplayBuffer(
            size, env.observation_space, env.action_space, policy_state
        )
        return cls(env_state, policy_state, callback_state, buffer)


class DQNState[PolicyType: AbstractQPolicy](AbstractAlgorithmState[PolicyType]):
    """
    Iteration-level state for DQN.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The step-level state.
        env: The environment being used.
        policy: The online policy being trained.
        opt_state: The optimizer state.
        callback_state: The callback state.
        target_policy: The target policy for stable Q-value estimation.
    """

    iteration_count: Int[Array, ""]
    step_state: DQNStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState
    target_policy: PolicyType


class DQN[PolicyType: AbstractQPolicy](
    AbstractAlgorithm[PolicyType, DQNState[PolicyType]]
):
    """
    Double Deep Q-Network (Double DQN) algorithm.

    Uses the online network to select actions and the target network to
    evaluate them, which reduces overestimation bias compared to standard DQN.

    The target network is a periodic copy of the online network, updated
    every ``target_update_interval`` iterations.

    Attributes:
        optimizer: The optimizer used for training.
        buffer_size: The size of the replay buffer.
        gamma: Discount factor for future rewards.
        learning_starts: Number of initial steps to collect before training.
        num_envs: Number of parallel environments.
        num_steps: Number of steps per iteration.
        batch_size: Batch size for training.
        target_update_interval: How often to copy the online network to the target.
        max_grad_norm: Maximum gradient norm for gradient clipping.

    Args:
        buffer_size: The size of the replay buffer.
        gamma: Discount factor for future rewards.
        learning_starts: Number of initial steps to collect before training.
        num_envs: Number of parallel environments.
        num_steps: Number of steps per iteration.
        batch_size: Batch size for training.
        target_update_interval: How often to copy the online network to the target.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        learning_rate: Learning rate for the optimizer.
    """

    optimizer: optax.GradientTransformation

    buffer_size: int
    gamma: float
    learning_starts: int

    num_envs: int
    num_steps: int
    batch_size: int

    target_update_interval: int
    max_grad_norm: float

    def __init__(
        self,
        *,
        buffer_size: int = 1_000_000,
        gamma: float = 0.99,
        learning_starts: int = 100,
        num_envs: int = 1,
        num_steps: int = 4,
        batch_size: int = 32,
        target_update_interval: int = 10_000,
        max_grad_norm: float = 10.0,
        learning_rate: optax.ScalarOrSchedule = 1e-4,
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.learning_starts = learning_starts

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm

        adam = optax.inject_hyperparams(optax.adam)(learning_rate)
        clip = optax.clip_by_global_norm(self.max_grad_norm)
        self.optimizer = optax.chain(clip, adam)

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: DQNStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> DQNStepState[PolicyType]:
        """Perform a single environment step and store in replay buffer."""
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

        return DQNStepState(
            next_env_state, next_policy_state, callback_state, replay_buffer
        )

    def collect_learning_starts(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: DQNStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> DQNStepState[PolicyType]:
        """Collect random initial experience before training begins."""

        def scan_step(
            carry: DQNStepState, key: Key[Array, ""]
        ) -> tuple[DQNStepState, None]:
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
        step_state: DQNStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> DQNStepState[PolicyType]:
        """Collect a rollout of experience into the replay buffer."""

        def scan_step(
            carry: DQNStepState, key: Key[Array, ""]
        ) -> tuple[DQNStepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return carry, None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.num_steps)
        )
        return step_state

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> DQNState[PolicyType]:
        init_key, starts_key, callback_key = jr.split(key, 3)

        if self.num_envs == 1:
            step_state = DQNStepState.initial(
                self.buffer_size, env, policy, callback, init_key
            )
            step_state = self.collect_learning_starts(
                env, policy, step_state, callback, starts_key
            )
        else:
            step_state = jax.vmap(
                DQNStepState.initial, in_axes=(None, None, None, None, 0)
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

        return DQNState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
            target_policy=policy,
        )

    def iteration(
        self,
        state: DQNState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> DQNState[PolicyType]:
        rollout_key, train_key, callback_key = jr.split(key, 3)

        if self.num_envs == 1:
            step_state = self.collect_rollout(
                state.env, state.policy, state.step_state, callback, rollout_key
            )
        else:
            step_state = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, eqx.if_array(0), None, 0)
            )(
                state.env,
                state.policy,
                state.step_state,
                callback,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, log = self.dqn_train(
            state.policy,
            state.opt_state,
            step_state.buffer,
            state.target_policy,
            key=train_key,
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
        state = state.with_callback_states(new_cb)
        return self.per_iteration(state)

    def per_iteration(self, state: DQNState[PolicyType]) -> DQNState[PolicyType]:
        """Periodically update the target network."""
        should_update = state.iteration_count % self.target_update_interval == 0
        target_policy = filter_cond(
            should_update,
            lambda: state.policy,
            lambda: state.target_policy,
        )
        return eqx.tree_at(lambda s: s.target_policy, state, target_policy)

    @staticmethod
    def dqn_loss(
        policy: PolicyType,
        batch: ReplayBuffer,
        target_policy: PolicyType,
        gamma: float,
    ) -> Float[Array, ""]:
        _, q_values = jax.vmap(policy.q_values)(batch.states, batch.observations)

        actions = batch.actions.astype(int)
        q_selected = q_values[jnp.arange(actions.shape[0]), actions]

        _, online_next_q = jax.vmap(policy.q_values)(
            batch.next_states, batch.next_observations
        )
        best_actions = jnp.argmax(online_next_q, axis=-1)

        _, target_next_q = jax.vmap(target_policy.q_values)(
            batch.next_states, batch.next_observations
        )
        next_q_selected = target_next_q[jnp.arange(actions.shape[0]), best_actions]

        not_terminal = (~batch.dones | batch.timeouts).astype(float)
        targets = batch.rewards + gamma * next_q_selected * not_terminal

        loss = jnp.mean(jnp.square(q_selected - targets)) / 2
        return loss

    dqn_loss_grad = staticmethod(eqx.filter_value_and_grad(dqn_loss))

    def dqn_train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        target_policy: PolicyType,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        batch = buffer.sample(self.batch_size, key=key)

        loss, grads = self.dqn_loss_grad(
            policy,
            batch,
            target_policy,
            self.gamma,
        )

        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, updates)

        log: dict[str, Scalar] = {"loss": loss}
        return policy, opt_state, log
