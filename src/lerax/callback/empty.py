from typing import final

from jax import numpy as jnp
from jaxtyping import Array, Bool, Key

from .base_callback import (
    AbstractStatelessCallback,
    EmptyCallbackState,
    EmptyCallbackStepState,
    IterationContext,
    StepContext,
    TrainingContext,
)


@final
class EmptyCallback(AbstractStatelessCallback):
    """A no-op callback that performs no actions."""

    def on_step(
        self, ctx: StepContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackStepState:
        return ctx.state

    def on_iteration(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        return ctx.state

    def on_training_start(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        return ctx.state

    def on_training_end(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        return ctx.state

    def continue_training(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return jnp.array(True)
