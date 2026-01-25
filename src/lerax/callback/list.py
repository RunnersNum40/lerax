import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Key

from .base_callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    IterationContext,
    ResetContext,
    StepContext,
    TrainingContext,
)


class CallbackListStepState(AbstractCallbackStepState):
    """
    State for CallbackList callback step.

    Attributes:
        states: List of step states for each callback.
    """

    states: list[AbstractCallbackStepState]


class CallbackListState(AbstractCallbackState):
    """
    State for CallbackList callback.

    Attributes:
        states: List of states for each callback.
    """

    states: list[AbstractCallbackState]


class CallbackList(AbstractCallback[CallbackListState, CallbackListStepState]):
    """
    Callback that aggregates multiple callbacks and forwards calls to them.

    Attributes:
        callbacks: List of callbacks to aggregate.

    Args:
        callbacks: List of callbacks to aggregate.
    """

    callbacks: list[AbstractCallback]

    def __init__(self, callbacks: list[AbstractCallback]) -> None:
        self.callbacks = callbacks

    def reset(self, ctx: ResetContext, *, key: Key[Array, ""]) -> CallbackListState:
        states = [
            callback.reset(ctx, key=key)
            for callback, key in zip(self.callbacks, jr.split(key, len(self.callbacks)))
        ]
        return CallbackListState(states=states)

    def step_reset(
        self, ctx: ResetContext, *, key: Key[Array, ""]
    ) -> CallbackListStepState:
        states = [
            callback.step_reset(ctx, key=key)
            for callback, key in zip(self.callbacks, jr.split(key, len(self.callbacks)))
        ]
        return CallbackListStepState(states=states)

    def on_step(
        self, ctx: StepContext, *, key: Key[Array, ""]
    ) -> CallbackListStepState:
        contexts = [
            eqx.tree_at(lambda c: c.state, ctx, state) for state in ctx.state.states
        ]
        new_states = [
            callback.on_step(ctx, key=key)
            for callback, ctx, key in zip(
                self.callbacks, contexts, jr.split(key, len(self.callbacks))
            )
        ]
        return CallbackListStepState(states=new_states)

    def on_iteration(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> CallbackListState:
        contexts = [
            eqx.tree_at(lambda c: (c.state, c.step_state), ctx, state)
            for state in zip(ctx.state.states, ctx.step_state.states)
        ]
        new_states = [
            callback.on_iteration(ctx, key=key)
            for callback, ctx, key in zip(
                self.callbacks, contexts, jr.split(key, len(self.callbacks))
            )
        ]
        return CallbackListState(states=new_states)

    def on_training_start(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> CallbackListState:
        contexts = [
            eqx.tree_at(lambda c: (c.state, c.step_state), ctx, state)
            for state in zip(ctx.state.states, ctx.step_state.states)
        ]
        new_states = [
            callback.on_training_start(ctx, key=key)
            for callback, ctx, key in zip(
                self.callbacks, contexts, jr.split(key, len(self.callbacks))
            )
        ]
        return CallbackListState(states=new_states)

    def on_training_end(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> CallbackListState:
        contexts = [
            eqx.tree_at(lambda c: (c.state, c.step_state), ctx, state)
            for state in zip(ctx.state.states, ctx.step_state.states)
        ]
        new_states = [
            callback.on_training_end(ctx, key=key)
            for callback, ctx, key in zip(
                self.callbacks, contexts, jr.split(key, len(self.callbacks))
            )
        ]
        return CallbackListState(states=new_states)

    def continue_training(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        contexts = [
            eqx.tree_at(lambda c: (c.state, c.step_state), ctx, state)
            for state in zip(ctx.state.states, ctx.step_state.states)
        ]
        continue_flags = [
            callback.continue_training(ctx, key=key)
            for callback, ctx, key in zip(
                self.callbacks, contexts, jr.split(key, len(self.callbacks))
            )
        ]
        return jnp.all(jnp.array(continue_flags))
