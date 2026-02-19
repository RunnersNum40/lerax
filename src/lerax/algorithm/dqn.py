from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.callback import AbstractCallback, IterationContext
from lerax.env import AbstractEnvLike
from lerax.policy import AbstractQPolicy
from lerax.utils import filter_cond

from .off_policy import (
    AbstractOffPolicyAlgorithm,
    AbstractOffPolicyState,
    AbstractOffPolicyStepState,
)


class DQNState[PolicyType: AbstractQPolicy](AbstractOffPolicyState[PolicyType]):
    """
    State for DQN algorithms.

    Extends the off-policy state with a target policy network that is
    periodically updated from the online policy.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The current step state.
        env: The environment being used.
        policy: The online policy being trained.
        opt_state: The optimizer state.
        callback_state: The callback state.
        target_policy: The target policy for stable Q-value estimation.
    """

    target_policy: PolicyType


class DQN[PolicyType: AbstractQPolicy](AbstractOffPolicyAlgorithm[PolicyType]):
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

    def per_step(
        self, step_state: AbstractOffPolicyStepState[PolicyType]
    ) -> AbstractOffPolicyStepState[PolicyType]:
        return step_state

    def per_iteration(
        self, state: AbstractOffPolicyState[PolicyType]
    ) -> AbstractOffPolicyState[PolicyType]:
        # Update target network periodically
        # state is always a DQNState at runtime
        should_update = state.iteration_count % self.target_update_interval == 0
        target_policy = filter_cond(
            should_update,
            lambda: state.policy,
            lambda: state.target_policy,  # type: ignore[attr-defined]
        )
        return eqx.tree_at(lambda s: s.target_policy, state, target_policy)

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> DQNState[PolicyType]:
        base_state = super().reset(env, policy, key=key, callback=callback)
        return DQNState(
            base_state.iteration_count,
            base_state.step_state,
            base_state.env,
            base_state.policy,
            base_state.opt_state,
            base_state.callback_state,
            target_policy=policy,
        )

    def iteration(
        self,
        state: AbstractOffPolicyState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> AbstractOffPolicyState[PolicyType]:
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
            state.target_policy,  # type: ignore[attr-defined]
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
                    locals(),
                ),
                key=callback_key,
            )
        )

        return self.per_iteration(state)

    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        # Not used directly; DQN overrides iteration() to call dqn_train instead.
        # This satisfies the abstract method requirement.
        return self.dqn_train(policy, opt_state, buffer, policy, key=key)

    @staticmethod
    def dqn_loss(
        policy: PolicyType,
        batch: ReplayBuffer,
        target_policy: PolicyType,
        gamma: float,
    ) -> Float[Array, ""]:
        # Compute Q-values for current observations
        _, q_values = jax.vmap(policy.q_values)(batch.states, batch.observations)

        # Select Q-values for taken actions
        actions = batch.actions.astype(int)
        q_selected = q_values[jnp.arange(actions.shape[0]), actions]

        # Double DQN: online network selects actions, target network evaluates
        _, online_next_q = jax.vmap(policy.q_values)(
            batch.next_states, batch.next_observations
        )
        best_actions = jnp.argmax(online_next_q, axis=-1)

        _, target_next_q = jax.vmap(target_policy.q_values)(
            batch.next_states, batch.next_observations
        )
        next_q_selected = target_next_q[jnp.arange(actions.shape[0]), best_actions]

        # Compute targets: only bootstrap for non-terminal transitions
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

        loss, grads = self.dqn_loss_grad(  # type: ignore[missing-argument]
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
