from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, final

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from lerax.callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    IterationContext,
    ResetContext,
    StepContext,
    TrainingContext,
)

if TYPE_CHECKING:
    from lerax.algorithm import AbstractAlgorithmState


@final
class AdaptiveCurriculumStepState(AbstractCallbackStepState):
    episode_metric: Float[Array, ""]


@final
class AdaptiveCurriculumState(AbstractCallbackState):
    level: Int[Array, ""]
    running_metric: Float[Array, ""]


class AbstractAdaptiveCurriculum(
    AbstractCallback[AdaptiveCurriculumState, AdaptiveCurriculumStepState]
):
    """
    Abstract base for adaptive curricula tracking a performance metric.

    Attributes:
        metric_fn: ``(done, reward, locals_dict) -> scalar`` function called each
            step and accumulated per episode.
        smoothing: Running-metric EMA factor; higher values favor recent episodes.

    """

    metric_fn: Callable = eqx.field(static=True)
    smoothing: float

    def reset(
        self, ctx: ResetContext, *, key: Key[Array, ""]
    ) -> AdaptiveCurriculumState:
        return AdaptiveCurriculumState(
            level=jnp.array(0, dtype=int),
            running_metric=jnp.array(0.0),
        )

    def step_reset(
        self, ctx: ResetContext, *, key: Key[Array, ""]
    ) -> AdaptiveCurriculumStepState:
        return AdaptiveCurriculumStepState(episode_metric=jnp.array(0.0))

    def on_step(
        self, ctx: StepContext, *, key: Key[Array, ""]
    ) -> AdaptiveCurriculumStepState:
        step_value = self.metric_fn(ctx.done, ctx.reward, ctx.locals)
        episode_metric = ctx.state.episode_metric + step_value
        episode_metric = jnp.where(ctx.done, jnp.array(0.0), episode_metric)
        return AdaptiveCurriculumStepState(episode_metric=episode_metric)

    def on_iteration(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> AdaptiveCurriculumState:
        step_metric = ctx.step_state.episode_metric
        running = ctx.state.running_metric
        updated_running = (1 - self.smoothing) * running + self.smoothing * step_metric
        return AdaptiveCurriculumState(
            level=ctx.state.level,
            running_metric=updated_running,
        )

    def on_training_start(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> AdaptiveCurriculumState:
        return ctx.state

    def on_training_end(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> AdaptiveCurriculumState:
        return ctx.state

    def continue_training(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return jnp.array(True)

    @abstractmethod
    def apply_curriculum[S: "AbstractAlgorithmState"](
        self, state: S, callback_state: AdaptiveCurriculumState
    ) -> tuple[S, AdaptiveCurriculumState]:
        """
        Modify algorithm state from the tracked metric.

        Called after ``on_iteration`` each iteration with the running metric and
        current level in ``callback_state``.
        """


@final
class LevelCurriculum(AbstractAdaptiveCurriculum):
    """
    Adapt an environment field through discrete parameter levels.

    Advance when the running performance metric exceeds ``threshold``.

    Args:
        where: Selector for the env field to modify.
        levels: Parameter values per level.
        metric_fn: Per-step metric extraction function.
        threshold: Advancement threshold.
        smoothing: EMA smoothing factor. Defaults to 0.05.

    Example::

        from lerax.curriculum import LevelCurriculum

        curriculum = LevelCurriculum(
            where=lambda env: env.max_speed,
            levels=jnp.array([4.0, 6.0, 8.0]),
            metric_fn=lambda done, reward, locals: reward,
            threshold=100.0,
        )
        algo.learn(env, policy, total_timesteps=..., key=key, callback=curriculum)
    """

    where: Callable = eqx.field(static=True)
    levels: Float[Array, " num_levels"]
    threshold: float

    def __init__(
        self,
        where: Callable,
        levels: Float[Array, " num_levels"],
        metric_fn: Callable,
        threshold: float,
        smoothing: float = 0.05,
    ):
        self.where = where
        self.levels = levels
        self.metric_fn = metric_fn
        self.threshold = threshold
        self.smoothing = smoothing

    def apply_curriculum[S: "AbstractAlgorithmState"](
        self, state: S, callback_state: AdaptiveCurriculumState
    ) -> tuple[S, AdaptiveCurriculumState]:
        current_level = callback_state.level
        metric = callback_state.running_metric

        should_advance = metric > self.threshold
        new_level = jnp.where(should_advance, current_level + 1, current_level)
        new_level = jnp.minimum(new_level, jnp.array(len(self.levels) - 1))

        new_param = self.levels[new_level]
        new_env = eqx.tree_at(self.where, state.env, new_param)
        state = eqx.tree_at(lambda s: s.env, state, new_env)

        new_callback_state = AdaptiveCurriculumState(
            level=new_level,
            running_metric=callback_state.running_metric,
        )
        return state, new_callback_state
