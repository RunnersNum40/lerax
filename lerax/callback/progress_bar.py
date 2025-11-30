from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array, Bool, Int, Key
from rich import progress, text

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy
from lerax.utils import callback_with_list_wrapper, callback_wrapper

from .base_callback import (
    AbstractCallback,
    AbstractCallbackStepState,
    EmptyCallbackState,
    IterationContext,
    ResetContext,
    StepContext,
    TrainingContext,
)


def superscript_digit(digit: int) -> str:
    return "⁰¹²³⁴⁵⁶⁷⁸⁹"[digit % 10]


def superscript_int(i: int) -> str:
    return "".join(superscript_digit(int(c)) for c in str(i))


def suffixes(base: int):
    yield ""

    val = 1
    while True:
        yield f"×{base}{superscript_int(val)}"
        val += 1


def unit_and_suffix(value: float, base: int) -> tuple[float, str]:
    if base < 1:
        raise ValueError("base must be >= 1")

    unit, suffix = 1, ""
    for i, suffix in enumerate(suffixes(base)):
        unit = base**i
        if int(value) < unit * base:
            break

    return unit, suffix


class SpeedColumn(progress.ProgressColumn):
    """
    Renders human readable speed.

    https://github.com/NichtJens/rich/tree/master
    """

    def render(self, task: progress.Task) -> text.Text:
        """Show speed."""
        speed = task.finished_speed or task.speed

        if speed is None:
            return text.Text("", style="progress.percentage")
        unit, suffix = unit_and_suffix(speed, 2)
        data_speed = speed / unit
        return text.Text(f"{data_speed:.1f}{suffix} it/s", style="red")


class JITProgressBar:
    progress_bar: progress.Progress
    task: progress.TaskID

    def __init__(self, name: str, total: int | None, transient: bool = False):
        self.progress_bar = progress.Progress(
            progress.TextColumn("[progress.description]{task.description}"),
            progress.SpinnerColumn(finished_text="[green]✔"),
            progress.MofNCompleteColumn(),
            progress.BarColumn(bar_width=None),
            progress.TaskProgressColumn(),
            progress.TextColumn("["),
            progress.TimeElapsedColumn(),
            progress.TextColumn("<"),
            progress.TimeRemainingColumn(),
            progress.TextColumn("]"),
            SpeedColumn(),
            transient=transient,
        )
        self.task = self.progress_bar.add_task(f"[yellow]{name}", total=total)

    @callback_wrapper
    def start(self) -> None:
        self.progress_bar.start()

    @callback_wrapper
    def stop(self) -> None:
        self.progress_bar.stop()

    @callback_with_list_wrapper
    def update(
        self, advance: float | None = None, completed: float | None = None
    ) -> None:
        self.progress_bar.update(self.task, advance=advance, completed=completed)


class ProgressBarCallbackStepState(AbstractCallbackStepState):
    steps: Int[Array, ""]


class ProgressBarCallback(
    AbstractCallback[EmptyCallbackState, ProgressBarCallbackStepState]
):
    """Callback for displaying a progress bar during training."""

    progress_bar: JITProgressBar

    def __init__(
        self,
        total_timesteps: int | None = None,
        name: str | None = None,
        env: AbstractEnvLike | None = None,
        policy: AbstractPolicy | None = None,
    ):
        if name is None:
            if env is not None:
                if policy is not None:
                    name = f"Training {policy.name} on {env.name}"
                else:
                    name = f"Training on {env.name}"
            else:
                if policy is not None:
                    name = f"Training {policy.name}"
                else:
                    name = "Training"
        else:
            name = name

        self.progress_bar = JITProgressBar(name, total=total_timesteps)

    def reset(self, ctx: ResetContext, *, key: Key) -> EmptyCallbackState:
        self.progress_bar.start()
        return EmptyCallbackState()

    def step_reset(
        self, ctx: ResetContext, *, key: Key
    ) -> ProgressBarCallbackStepState:
        return ProgressBarCallbackStepState(jnp.array(0, dtype=int))

    def on_step(self, ctx: StepContext, *, key: Key) -> ProgressBarCallbackStepState:
        return ProgressBarCallbackStepState(ctx.state.steps + 1)

    def on_iteration(self, ctx: IterationContext, *, key: Key) -> EmptyCallbackState:
        self.progress_bar.update(completed=jnp.sum(ctx.step_state.steps))
        return ctx.state

    def on_training_start(
        self, ctx: TrainingContext, *, key: Key
    ) -> EmptyCallbackState:
        return ctx.state

    def on_training_end(self, ctx: TrainingContext, *, key: Key) -> EmptyCallbackState:
        # TODO: Fix ordered callback issue
        # self.progress_bar.stop()
        return ctx.state

    def continue_training(self, ctx: IterationContext, *, key: Key) -> Bool[Array, ""]:
        return jnp.array(True)
