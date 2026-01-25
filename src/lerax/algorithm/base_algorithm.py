from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import equinox as eqx
import optax
from jax import random as jr
from jaxtyping import Array, Int, Key

from lerax.callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    CallbackList,
    TrainingContext,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicy, AbstractPolicyState
from lerax.utils import filter_scan


class AbstractStepState(eqx.Module):
    """
    Base class for algorithm state that is vectorized over environment steps.

    Attributes:
        env_state: The state of the environment.
        policy_state: The state of the policy.
        callback_state: The state of the callback for this step.
    """

    env_state: eqx.AbstractVar[AbstractEnvLikeState]
    policy_state: eqx.AbstractVar[AbstractPolicyState]
    callback_state: eqx.AbstractVar[AbstractCallbackStepState]

    def with_callback_state[A: AbstractStepState](
        self: A, callback_state: AbstractCallbackStepState | None
    ) -> A:
        """
        Return a new step state with the given callback state.

        Args:
            callback_state: The new callback state. If None, the existing state is used.

        Returns:
            A new step state with the updated callback state.
        """
        return eqx.tree_at(lambda s: s.callback_state, self, callback_state)


class AbstractAlgorithmState[PolicyType: AbstractPolicy](eqx.Module):
    """
    Base class for algorithm states.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The state for the current step.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The optimizer state.
        callback_state: The state of the callback for this iteration.
    """

    iteration_count: eqx.AbstractVar[Int[Array, ""]]
    step_state: eqx.AbstractVar[AbstractStepState]
    env: eqx.AbstractVar[AbstractEnvLike]
    policy: eqx.AbstractVar[PolicyType]
    opt_state: eqx.AbstractVar[optax.OptState]
    callback_state: eqx.AbstractVar[AbstractCallbackState]

    def next[A: AbstractAlgorithmState](
        self: A,
        step_state: AbstractStepState,
        policy: PolicyType,
        opt_state: optax.OptState,
    ) -> A:
        """
        Return a new algorithm state for the next iteration.

        Increments the iteration count and updates the step state, policy, and
        optimizer state.

        Args:
            step_state: The new step state.
            policy: The new policy.
            opt_state: The new optimizer state.

        Returns:
            A new algorithm state with the updated fields.
        """
        return eqx.tree_at(
            lambda s: (s.iteration_count, s.step_state, s.policy, s.opt_state),
            self,
            (self.iteration_count + 1, step_state, policy, opt_state),
        )

    def with_callback_states[A: AbstractAlgorithmState](
        self: A, callback_state: AbstractCallbackState
    ) -> A:
        """
        Return a new algorithm state with the given callback state.

        Args:
            callback_state: The new callback state.

        Returns:
            A new algorithm state with the updated callback state.
        """
        return eqx.tree_at(lambda s: s.callback_state, self, callback_state)


class AbstractAlgorithm[PolicyType: AbstractPolicy, StateType: AbstractAlgorithmState](
    eqx.Module
):
    """
    Base class for RL algorithms.

    Provides the main training loop and abstract methods for algorithm-specific behavior.

    Attributes:
        optimizer: The optimizer used for training.
        num_envs: The number of parallel environments.
        num_steps: The number of steps per environment per iteration.
    """

    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]

    @abstractmethod
    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> StateType:
        """
        Return the initial carry for the training iteration.

        Args:
            env: The environment to train on.
            policy: The policy to train.
            key: A JAX PRNG key.
            callback: A callback or list of callbacks to use during training.

        Returns:
            The initial algorithm state.
        """

    @abstractmethod
    def per_iteration(self, state: StateType) -> StateType:
        """
        Process the algorithm state after each iteration.

        Used for algorithm-specific bookkeeping.

        Args:
            state: The current algorithm state.

        Returns:
            The updated algorithm state.
        """

    @abstractmethod
    def iteration(
        self,
        state: StateType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> StateType:
        """
        Perform a single iteration of training.

        Args:
            state: The current algorithm state.
            key: A JAX PRNG key.
            callback: A callback or list of callbacks to use during training.

        Returns:
            The updated algorithm state.
        """

    @abstractmethod
    def num_iterations(self, total_timesteps: int) -> int:
        """Number of iterations per training session."""

    @eqx.filter_jit
    def learn(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        total_timesteps: int,
        *,
        key: Key[Array, ""],
        callback: Sequence[AbstractCallback] | AbstractCallback | None = None,
    ) -> PolicyType:
        """
        Train the policy on the environment for a given number of timesteps.

        Args:
            env: The environment to train on.
            policy: The policy to train.
            total_timesteps: The total number of timesteps to train for.
            key: A JAX PRNG key.
            callback: A callback or list of callbacks to use during training.

        Returns:
            policy: The trained policy.
        """
        callback_start_key, reset_key, learn_key, callback_end_key = jr.split(key, 4)

        if callback is None:
            callback = CallbackList(callbacks=[])
        elif isinstance(callback, Sequence):
            callback = CallbackList(callbacks=list(callback))
        elif isinstance(callback, AbstractCallback):
            callback = callback
        else:
            raise TypeError(f"Invalid callback type: {type(callback)}")

        state = self.reset(env, policy, key=reset_key, callback=callback)

        state = state.with_callback_states(
            callback.on_training_start(
                ctx=TrainingContext(
                    state.callback_state,
                    state.step_state.callback_state,
                    env,
                    state.policy,
                    total_timesteps,
                    state.iteration_count,
                    state.opt_state,
                    locals(),
                ),
                key=callback_start_key,
            )
        )

        state, _ = filter_scan(
            lambda s, k: (self.iteration(s, key=k, callback=callback), None),
            state,
            jr.split(learn_key, self.num_iterations(total_timesteps)),
        )

        state = state.with_callback_states(
            callback.on_training_end(
                ctx=TrainingContext(
                    state.callback_state,
                    state.step_state.callback_state,
                    env,
                    state.policy,
                    total_timesteps,
                    state.iteration_count,
                    state.opt_state,
                    locals(),
                ),
                key=callback_end_key,
            )
        )

        return state.policy
