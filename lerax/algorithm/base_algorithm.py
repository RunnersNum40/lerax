from __future__ import annotations

from abc import abstractmethod
from typing import Self, Sequence, cast

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
from lerax.policy import AbstractPolicy, AbstractPolicyState, AbstractStatefulPolicy
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

    def with_callback_state(
        self, callback_state: AbstractCallbackStepState | None
    ) -> Self:
        """
        Return a new step state with the given callback state.

        If callback_state is None, the existing state is kept.

        Args:
            callback_state: The new callback state, or None to keep existing.

        Returns:
            A new step state with the updated callback state.
        """
        if callback_state is None:
            return self
        return eqx.tree_at(lambda s: s.callback_state, self, callback_state)


class AbstractAlgorithmState[PolicyType: AbstractStatefulPolicy](eqx.Module):
    """
    Base class for algorithm states.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The state for the current step.
        policy: The policy being trained.
        opt_state: The optimizer state.
        callback_state: The state of the callback for this iteration.
    """

    iteration_count: eqx.AbstractVar[Int[Array, ""]]
    step_state: eqx.AbstractVar[AbstractStepState]
    policy: eqx.AbstractVar[PolicyType]
    opt_state: eqx.AbstractVar[optax.OptState]
    callback_state: eqx.AbstractVar[AbstractCallbackState]

    def next(
        self,
        step_state: AbstractStepState,
        policy: PolicyType,
        opt_state: optax.OptState,
    ) -> Self:
        """
        Return a new algorithm state for the next iteration.

        Increments the iteration count and updates the step state, policy, and
        optimizer state.
        """
        return eqx.tree_at(
            lambda s: (s.iteration_count, s.step_state, s.policy, s.opt_state),
            self,
            (self.iteration_count + 1, step_state, policy, opt_state),
        )

    def with_callback_state(self, callback_state: AbstractCallbackState) -> Self:
        """
        Return a new algorithm state with the given callback state.
        """
        return eqx.tree_at(lambda s: s.callback_state, self, callback_state)


class AbstractAlgorithmRunner[
    PolicyType: AbstractStatefulPolicy,
    StateType: AbstractAlgorithmState,
](eqx.Module):
    """
    Base class for fully configured RL algorithm runners.

    A runner has all runtime parameters bound (environment, callbacks, optimizer,
    horizon) and exposes a `__call__` method taking only a policy and a key.
    """

    env: eqx.AbstractVar[AbstractEnvLike]
    callback: eqx.AbstractVar[AbstractCallback]
    optimizer: eqx.AbstractVar[optax.GradientTransformation]

    num_envs: eqx.AbstractVar[int]
    num_steps: eqx.AbstractVar[int]
    total_timesteps: eqx.AbstractVar[int]
    num_iterations: eqx.AbstractVar[int]

    @abstractmethod
    def reset(
        self,
        policy: PolicyType,
        *,
        key: Key,
    ) -> StateType:
        """
        Return the initial algorithm state.

        Responsible for resetting environment, policy, callback state, and
        initializing the optimizer state.
        """

    @abstractmethod
    def iteration(
        self,
        state: StateType,
        *,
        key: Key,
    ) -> StateType:
        """
        Perform a single training iteration.

        Typically:
          - collect rollout / samples
          - optimize policy
          - update state (including iteration_count)
          - fire iteration callbacks
        """

    @abstractmethod
    def per_iteration(self, state: StateType) -> StateType:
        """
        Optional hook to process state after each iteration.

        Concrete runners can use this for algorithm-specific bookkeeping
        (e.g. target network updates). The base implementation is never called.
        """

    def init_stateful_policy(
        self,
        policy: AbstractPolicy,
    ) -> tuple[PolicyType, bool]:
        """
        Convert a policy to a stateful policy if needed.

        Returns:
            (stateful_policy, was_stateless)
        """
        if isinstance(policy, AbstractStatefulPolicy):
            stateful_policy = cast(PolicyType, policy)
            return stateful_policy, False
        stateful_policy = cast(PolicyType, policy.into_stateful())
        return stateful_policy, True

    @eqx.filter_jit
    def __call__[P: AbstractPolicy](
        self,
        policy: P,
        *,
        key: Key,
    ) -> P:
        """
        Run training.

        Args:
            policy: The initial policy.
            key: A JAX PRNG key.

        Returns:
            The trained policy.
        """
        callback = self.callback
        env = self.env
        total_timesteps = self.total_timesteps

        callback_start_key, reset_key, learn_key, callback_end_key = jr.split(key, 4)

        stateful_policy, was_stateless = self.init_stateful_policy(policy)

        state = self.reset(stateful_policy, key=reset_key)

        state = state.with_callback_state(
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

        def scan_body(
            carry: StateType,
            iter_key: Key,
        ) -> tuple[StateType, None]:
            new_state = self.iteration(carry, key=iter_key)
            return new_state, None

        state, _ = filter_scan(
            scan_body,
            state,
            jr.split(learn_key, self.num_iterations),
        )

        state = state.with_callback_state(
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

        if was_stateless:
            return state.policy.into_stateless()
        else:
            return state.policy


class AbstractAlgorithm[RunnerType: AbstractAlgorithmRunner](eqx.Module):
    """
    Base class for RL algorithm builders.

    A builder is an immutable configuration object with fluent `with_*` methods.
    It binds:
      - hyperparameters (in concrete subclasses)
      - num_envs
      - env
      - total_timesteps
      - callback

    and produces a fully configured runner via `build()`. The convenience
    method `learn` delegates to the runner.
    """

    num_envs: eqx.AbstractVar[int]
    env: eqx.AbstractVar[AbstractEnvLike]
    total_timesteps: eqx.AbstractVar[int]
    callback: eqx.AbstractVar[AbstractCallback]

    def with_num_envs(self, num_envs: int) -> Self:
        """
        Return a new builder with the given number of environments.
        """
        return eqx.tree_at(lambda a: a.num_envs, self, num_envs)

    def with_env(self, env: AbstractEnvLike) -> Self:
        """
        Return a new builder with the given environment bound.
        """
        return eqx.tree_at(lambda a: a.env, self, env)

    def with_total_timesteps(self, total_timesteps: int) -> Self:
        """
        Return a new builder with the given total timesteps bound.
        """
        return eqx.tree_at(lambda a: a.total_timesteps, self, total_timesteps)

    def with_callbacks(
        self,
        callback: AbstractCallback | Sequence[AbstractCallback] | None,
    ) -> Self:
        """
        Return a new builder with the given callback(s) bound.

        Normalizes:
          - None -> empty CallbackList
          - single AbstractCallback -> as-is
          - sequence of callbacks -> CallbackList
        """
        if callback is None:
            callback = CallbackList(callbacks=[])
        elif isinstance(callback, AbstractCallback):
            callback = callback
        else:
            callback = CallbackList(callbacks=list(callback))

        return eqx.tree_at(lambda a: a.callback, self, callback)

    def build(self) -> RunnerType:
        """
        Build a fully configured algorithm runner.

        Raises:
            ValueError: if env or total_timesteps are not set.
        """
        if self.env is None:
            raise ValueError("Environment must be set on the builder before build().")
        if self.total_timesteps is None:
            raise ValueError(
                "total_timesteps must be set on the builder before build()."
            )

        return self.build_runner(self.env, self.total_timesteps, self.callback)

    @abstractmethod
    def build_runner(
        self,
        env: AbstractEnvLike,
        total_timesteps: int,
        callback: AbstractCallback,
    ) -> RunnerType:
        """
        Construct the concrete runner.

        Concrete subclasses are responsible for:
          - computing derived quantities (num_iterations, batch sizes, etc.)
          - constructing the optimizer (including schedules)
          - populating all fields on the runner
        """

    def learn(
        self,
        policy: AbstractPolicy,
        key: Key,
    ) -> AbstractPolicy:
        """
        Train a policy using the provided algorithm configuration.

        Args:
            policy: The initial policy.
            key: A JAX PRNG key.

        Returns:
            A trained policy.
        """
        runner = self.build()
        return runner(policy, key=key)
