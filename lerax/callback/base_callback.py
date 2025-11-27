from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Bool, Key


class AbstractCallbackState(eqx.Module):
    """Base class for callback states."""


class AbstractCallback[StateType: AbstractCallbackState](eqx.Module):
    """
    Base class for RL algorithm callbacks.

    Should be subclassed to create custom callbacks. All concrete methods should
    work under JIT compilation.
    """

    @abstractmethod
    def reset(self, locals, *, key: Key) -> StateType:
        """Initialize the callback state."""


class EmptyCallbackState(AbstractCallbackState):
    """Empty state for EmptyCallback."""


class AbstractStatelessCallback(AbstractCallback[EmptyCallbackState]):
    def reset(self, locals, *, key: Key) -> EmptyCallbackState:
        return EmptyCallbackState()


class AbstractCallbackStepState(eqx.Module):
    """
    Base class for callback states that are vectorized across environment
    steps.
    """


class AbstractVectorizedCallback[StateType: AbstractCallbackState](
    AbstractCallback[StateType]
):
    """
    Base class for callbacks that are vectorized across environment steps.
    """

    @abstractmethod
    def step_reset(self, locals, *, key: Key) -> AbstractCallbackStepState:
        """Reset the callback state for vectorized steps."""


class AbstractStepCallback[
    StateType: AbstractCallbackState, StepStateType: AbstractCallbackStepState
](AbstractVectorizedCallback[StateType]):
    """
    Callback that implements step-related methods.

    Step-related methods are vectorized across multiple environments.
    """

    @abstractmethod
    def on_step_start(self, state: StepStateType, locals, *, key: Key) -> StepStateType:
        """Called at the start of each environment step."""

    @abstractmethod
    def on_step_end(self, state: StepStateType, locals, *, key: Key) -> StepStateType:
        """Called at the end of each environment step."""


class AbstractIterationCallback[StateType: AbstractCallbackState](
    AbstractCallback[StateType]
):
    """Callback that implements iteration-related methods."""

    @abstractmethod
    def on_iteration_start(self, state: StateType, locals, *, key: Key) -> StateType:
        """Called at the start of each training iteration."""

    @abstractmethod
    def on_iteration_end(self, state: StateType, locals, *, key: Key) -> StateType:
        """Called at the end of each training iteration."""


class AbstractTrainingCallback[StateType: AbstractCallbackState](
    AbstractCallback[StateType]
):
    """Callback that implements training-related methods."""

    @abstractmethod
    def on_training_start(self, state: StateType, locals, *, key: Key) -> StateType:
        """Called at the start of training."""

    @abstractmethod
    def on_training_end(self, state: StateType, locals, *, key: Key) -> StateType:
        """Called at the end of training."""


class AbstractContinueTrainingCallback[StateType: AbstractCallbackState](
    AbstractCallback[StateType]
):
    @abstractmethod
    def continue_training(
        self, state: StateType, locals, *, key: Key
    ) -> Bool[Array, ""]:
        """Called at the end of each iteration to determine whether to continue."""
