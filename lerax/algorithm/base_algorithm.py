from __future__ import annotations

from abc import abstractmethod
from typing import Sequence, cast

import equinox as eqx
import optax
from jax import random as jr
from jaxtyping import Array, Int, Key

from lerax.callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    AbstractTrainingCallback,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicy, AbstractPolicyState, AbstractStatefulPolicy
from lerax.utils import filter_scan


class AbstractStepState(eqx.Module):
    """Base class for algorithm state that is vectorized over environment steps."""

    env_state: eqx.AbstractVar[AbstractEnvLikeState]
    policy_state: eqx.AbstractVar[AbstractPolicyState]
    callback_states: eqx.AbstractVar[list[AbstractCallbackStepState | None]]

    def with_callback_state[A: AbstractStepState](
        self: A, callback_state: AbstractCallbackStepState | None
    ) -> A:
        return eqx.tree_at(lambda s: s.callback_state, self, callback_state)


class AbstractAlgorithmState[PolicyType: AbstractStatefulPolicy](eqx.Module):
    """Base class for algorithm states."""

    iteration_count: eqx.AbstractVar[Int[Array, ""]]
    step_state: eqx.AbstractVar[AbstractStepState]
    env: eqx.AbstractVar[AbstractEnvLike]
    policy: eqx.AbstractVar[PolicyType]
    opt_state: eqx.AbstractVar[optax.OptState]
    callback_states: eqx.AbstractVar[list[AbstractCallbackState]]

    def next[A: AbstractAlgorithmState](
        self: A,
        step_state: AbstractStepState,
        policy: PolicyType,
        opt_state: optax.OptState,
    ) -> A:
        return eqx.tree_at(
            lambda s: (s.iteration_count, s.step_state, s.policy, s.opt_state),
            self,
            (self.iteration_count + 1, step_state, policy, opt_state),
        )

    def with_callback_states[A: AbstractAlgorithmState](
        self: A, callback_states: list[AbstractCallbackState]
    ) -> A:
        return eqx.tree_at(lambda s: s.callback_states, self, callback_states)


class AbstractAlgorithm[
    PolicyType: AbstractStatefulPolicy, StateType: AbstractAlgorithmState
](eqx.Module):
    """Base class for RL algorithms."""

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]

    @abstractmethod
    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key,
        callbacks: list[AbstractCallback],
    ) -> StateType:
        """Return the initial carry for the training iteration."""

    @abstractmethod
    def per_iteration(self, state: StateType) -> StateType:
        """Process the algorithm state after each iteration."""

    @abstractmethod
    def iteration(
        self,
        state: StateType,
        *,
        key: Key,
        callbacks: list[AbstractCallback],
    ) -> StateType:
        """
        Perform a single iteration of training.

        **Arguments:**
            - state: The current algorithm state.
            - key: A JAX Key.
            - callbacks: A list of training callbacks.

        **Returns:**
            - state: The updated algorithm state.
        """

    @abstractmethod
    def num_iterations(self, total_timesteps: int) -> int:
        """Number of iterations per training session."""

    @eqx.filter_jit
    def learn[A: AbstractPolicy](
        self,
        env: AbstractEnvLike,
        policy: A,
        total_timesteps: int,
        *,
        key: Key,
        callbacks: Sequence[AbstractCallback] | AbstractCallback | None = None,
    ) -> A:
        """
        Train the policy on the environment for a given number of timesteps.

        **Arguments:**
            - env: The environment to train on.
            - policy: The policy to train.
            - total_timesteps: The total number of timesteps to train for.
            - key: A JAX Key.
            - callbacks: A sequence of training callback, a single callback, or None.

        **Returns:**
            - policy: The trained policy.
        """
        callback_start_key, reset_key, learn_key, callback_end_key = jr.split(key, 4)

        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, AbstractCallback):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)

        if isinstance(policy, AbstractStatefulPolicy):
            _policy = cast(PolicyType, policy)
        else:
            _policy = cast(PolicyType, policy.into_stateful())

        state = self.reset(env, _policy, key=reset_key, callbacks=callbacks)

        callback_states = [
            (
                callback.on_training_start(
                    callback_state, locals(), key=callback_start_key
                )
                if isinstance(callback, AbstractTrainingCallback)
                else callback_state
            )
            for callback_state, callback in zip(state.callback_states, callbacks)
        ]
        state = state.with_callback_states(callback_states)

        state, _ = filter_scan(
            lambda s, k: (self.iteration(s, key=k, callbacks=callbacks), None),
            state,
            jr.split(learn_key, self.num_iterations(total_timesteps)),
        )

        [
            (
                callback.on_training_end(callback_state, locals(), key=callback_end_key)
                if isinstance(callback, AbstractTrainingCallback)
                else callback_state
            )
            for callback_state, callback in zip(state.callback_states, callbacks)
        ]

        if isinstance(policy, AbstractStatefulPolicy):
            return state.policy
        else:
            return state.policy.into_stateless()
