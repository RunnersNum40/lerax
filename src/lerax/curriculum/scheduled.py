"""Scheduled curriculum callbacks that modify environment parameters over training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Key

from lerax.callback import (
    AbstractStatelessCallback,
    IterationContext,
    StepContext,
)
from lerax.callback.base_callback import EmptyCallbackState

if TYPE_CHECKING:
    from lerax.algorithm import AbstractAlgorithmState


def linear_schedule(
    start: float, end: float, total: int
) -> Callable[[Int[Array, ""]], Float[Array, ""]]:
    """
    Linear interpolation from ``start`` to ``end`` over ``total`` iterations.

    Clamps to ``[start, end]`` outside the range.

    Args:
        start: Value at iteration 0.
        end: Value at iteration ``total``.
        total: Number of iterations for the full transition.

    Returns:
        A function mapping iteration count to the scheduled value.
    """
    start_arr = jnp.array(start, dtype=float)
    end_arr = jnp.array(end, dtype=float)

    def schedule(iteration: Int[Array, ""]) -> Float[Array, ""]:
        t = jnp.clip(iteration / total, 0.0, 1.0)
        return start_arr + (end_arr - start_arr) * t

    return schedule


def step_schedule(
    values: list[float], boundaries: list[int]
) -> Callable[[Int[Array, ""]], Float[Array, ""]]:
    """
    Step-wise schedule that jumps between discrete values at specified
    iteration boundaries.

    Args:
        values: Parameter values for each stage. Length must be
            ``len(boundaries) + 1``.
        boundaries: Iteration counts at which to transition to the next
            value.

    Returns:
        A function mapping iteration count to the scheduled value.
    """
    values_arr = jnp.array(values, dtype=float)
    boundaries_arr = jnp.array(boundaries, dtype=int)

    def schedule(iteration: Int[Array, ""]) -> Float[Array, ""]:
        idx = jnp.searchsorted(boundaries_arr, iteration, side="right")
        return values_arr[idx]

    return schedule


def cosine_schedule(
    start: float, end: float, total: int
) -> Callable[[Int[Array, ""]], Float[Array, ""]]:
    """
    Cosine annealing from ``start`` to ``end`` over ``total`` iterations.

    Args:
        start: Value at iteration 0.
        end: Value at iteration ``total``.
        total: Number of iterations for the full transition.

    Returns:
        A function mapping iteration count to the scheduled value.
    """
    start_arr = jnp.array(start, dtype=float)
    end_arr = jnp.array(end, dtype=float)

    def schedule(iteration: Int[Array, ""]) -> Float[Array, ""]:
        t = jnp.clip(iteration / total, 0.0, 1.0)
        return end_arr + (start_arr - end_arr) * (1 + jnp.cos(jnp.pi * t)) / 2

    return schedule


class ScheduledCurriculum(AbstractStatelessCallback):
    """
    Curriculum callback that modifies an environment field on a fixed
    schedule.

    Uses ``eqx.tree_at`` to update a field on ``state.env`` each
    iteration based on the current iteration count.

    Multiple ``ScheduledCurriculum`` instances can be composed via
    ``CallbackList`` to schedule multiple fields simultaneously.

    Attributes:
        where: A function selecting the field to modify on the env,
            e.g. ``lambda env: env.mass``.
        schedule_fn: A function mapping iteration count to the
            scheduled parameter value.

    Args:
        where: Selector for the env field to schedule.
        schedule_fn: Schedule function (see ``linear_schedule``,
            ``step_schedule``, ``cosine_schedule``).

    Example::

        from lerax.curriculum import ScheduledCurriculum, linear_schedule

        curriculum = ScheduledCurriculum(
            where=lambda env: env.m,
            schedule_fn=linear_schedule(start=0.5, end=2.0, total=1000),
        )
        algo.learn(env, policy, total_timesteps=..., key=key, callback=curriculum)
    """

    where: Callable = eqx.field(static=True)
    schedule_fn: Callable = eqx.field(static=True)

    def on_step(self, ctx: StepContext, *, key: Key[Array, ""]):
        return ctx.state

    def on_iteration(self, ctx: IterationContext, *, key: Key[Array, ""]):
        return ctx.state

    def on_training_start(self, ctx, *, key: Key[Array, ""]):
        return ctx.state

    def on_training_end(self, ctx, *, key: Key[Array, ""]):
        return ctx.state

    def continue_training(self, ctx: IterationContext, *, key: Key[Array, ""]):
        return jnp.array(True)

    def apply_curriculum[S: "AbstractAlgorithmState"](
        self, state: S, callback_state: EmptyCallbackState
    ) -> tuple[S, EmptyCallbackState]:
        new_val = self.schedule_fn(state.iteration_count)
        new_env = eqx.tree_at(self.where, state.env, new_val)
        return eqx.tree_at(lambda s: s.env, state, new_env), callback_state
