from types import SimpleNamespace
from typing import cast

from jax import random as jr

from lerax.callback import EmptyCallback
from lerax.callback.base_callback import (
    EmptyCallbackState,
    EmptyCallbackStepState,
    IterationContext,
    ResetContext,
    StepContext,
    TrainingContext,
)


def test_empty_callback_preserves_empty_state():
    callback = EmptyCallback()
    key = jr.key(0)
    callback_state = callback.reset(ResetContext(locals={}), key=key)
    callback_step_state = callback.step_reset(ResetContext(locals={}), key=key)
    step_context = cast(StepContext, SimpleNamespace(state=callback_step_state))
    iteration_context = cast(
        IterationContext,
        SimpleNamespace(state=callback_state),
    )
    training_context = cast(
        TrainingContext,
        SimpleNamespace(state=callback_state),
    )

    assert isinstance(callback_state, EmptyCallbackState)
    assert isinstance(callback_step_state, EmptyCallbackStepState)
    assert callback.on_step(step_context, key=key) is callback_step_state
    assert callback.on_iteration(iteration_context, key=key) is callback_state
    assert callback.on_training_start(training_context, key=key) is callback_state
    assert callback.on_training_end(training_context, key=key) is callback_state
    assert callback.continue_training(iteration_context, key=key)
