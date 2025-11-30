from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import optax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy


class AbstractCallbackStepState(eqx.Module):
    """
    Base class for callback states that are vectorized across environment
    steps.
    """


class AbstractCallbackState(eqx.Module):
    """Base class for callback states."""


class ResetContext(eqx.Module):
    """Context passed to the reset method of callbacks."""

    locals: dict


class StepContext[StepStateType: AbstractCallbackStepState](eqx.Module):
    """Values passed to step-related callback methods."""

    state: StepStateType
    env: AbstractEnvLike
    policy: AbstractPolicy
    done: Bool[Array, ""]
    reward: Float[Array, ""]
    locals: dict


class IterationContext[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](eqx.Module):
    """Values passed to iteration-related callback methods."""

    state: StateType
    step_state: StepStateType
    env: AbstractEnvLike
    policy: AbstractPolicy
    iteration_count: Int[Array, ""]
    opt_state: optax.OptState
    training_log: dict[str, Array]
    locals: dict


class TrainingContext[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](eqx.Module):
    """Values passed to training-related callback methods."""

    state: StateType
    step_state: StepStateType
    env: AbstractEnvLike
    policy: AbstractPolicy
    total_timesteps: int
    iteration_count: Int[Array, ""]
    opt_state: optax.OptState
    locals: dict


class AbstractCallback[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](eqx.Module):
    """
    Base class for RL algorithm callbacks.

    Should be subclassed to create custom callbacks. All concrete methods should
    work under JIT compilation.
    """

    @abstractmethod
    def reset(self, ctx: ResetContext, *, key: Key) -> StateType:
        """Initialize the callback state."""

    @abstractmethod
    def step_reset(self, ctx: ResetContext, *, key: Key) -> StepStateType:
        """Reset the callback state for vectorized steps."""

    @abstractmethod
    def on_step(self, ctx: StepContext, *, key: Key) -> StepStateType:
        """Called at the end of each environment step."""

    @abstractmethod
    def on_iteration(self, ctx: IterationContext, *, key: Key) -> StateType:
        """Called at the end of each training iteration."""

    @abstractmethod
    def on_training_start(self, ctx: TrainingContext, *, key: Key) -> StateType:
        """Called at the start of training."""

    @abstractmethod
    def on_training_end(self, ctx: TrainingContext, *, key: Key) -> StateType:
        """Called at the end of training."""

    @abstractmethod
    def continue_training(self, ctx: IterationContext, *, key: Key) -> Bool[Array, ""]:
        """Called at the end of each iteration to determine whether to continue."""


class EmptyCallbackStepState(AbstractCallbackStepState):
    """Empty step state for stateless callbacks."""


class EmptyCallbackState(AbstractCallbackState):
    """Empty state for stateless callbacks."""


class AbstractStatelessCallback(
    AbstractCallback[EmptyCallbackState, EmptyCallbackStepState]
):
    """Callback that does not maintain any state."""

    def reset(self, ctx: ResetContext, *, key: Key) -> EmptyCallbackState:
        return EmptyCallbackState()

    def step_reset(self, ctx: ResetContext, *, key: Key) -> EmptyCallbackStepState:
        return EmptyCallbackStepState()


class AbstractStepCallback[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](AbstractCallback[StateType, StepStateType]):
    """Callback that implements step-related methods."""

    def on_iteration(self, ctx: IterationContext, *, key: Key) -> StateType:
        return ctx.state

    def on_training_start(self, ctx: TrainingContext, *, key: Key) -> StateType:
        return ctx.state

    def on_training_end(self, ctx: TrainingContext, *, key: Key) -> StateType:
        return ctx.state

    def continue_training(self, ctx: IterationContext, *, key: Key) -> Bool[Array, ""]:
        return jnp.array(True)


class AbstractIterationCallback[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](AbstractCallback[StateType, StepStateType]):
    """Callback that implements iteration-related methods."""

    def on_step(self, ctx: StepContext, *, key: Key) -> StepStateType:
        return ctx.state

    def on_training_start(self, ctx: TrainingContext, *, key: Key) -> StateType:
        return ctx.state

    def on_training_end(self, ctx: TrainingContext, *, key: Key) -> StateType:
        return ctx.state

    def continue_training(self, ctx: IterationContext, *, key: Key) -> Bool[Array, ""]:
        return jnp.array(True)


class AbstractTrainingCallback[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](AbstractCallback[StateType, StepStateType]):
    """Callback that implements training-related methods."""

    def on_step(self, ctx: StepContext, *, key: Key) -> StepStateType:
        return ctx.state

    def on_iteration(self, ctx: IterationContext, *, key: Key) -> StateType:
        return ctx.state

    def continue_training(self, ctx: IterationContext, *, key: Key) -> Bool[Array, ""]:
        return jnp.array(True)


class AbstractContinueTrainingCallback[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](AbstractCallback[StateType, StepStateType]):
    """Callback that implements continue training method."""

    def on_step(self, ctx: StepContext, *, key: Key) -> StepStateType:
        return ctx.state

    def on_iteration(self, ctx: IterationContext, *, key: Key) -> StateType:
        return ctx.state

    def on_training_start(self, ctx: TrainingContext, *, key: Key) -> StateType:
        return ctx.state

    def on_training_end(self, ctx: TrainingContext, *, key: Key) -> StateType:
        return ctx.state
