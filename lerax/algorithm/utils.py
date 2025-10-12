from __future__ import annotations

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Scalar, ScalarLike
from rich import progress, text
from tensorboardX import SummaryWriter

from lerax.utils import (
    debug_with_list_wrapper,
    debug_with_numpy_wrapper,
    debug_wrapper,
)


class EpisodeStatisticsAccumulator(eqx.Module):
    """
    Accumulator for episode statistics.
    This is used to keep track of the episode length, reward, and done status.
    """

    episode_length: Int[Array, ""]
    episode_reward: Float[Array, ""]
    episode_done: Bool[Array, ""]

    @classmethod
    def from_episode_stats(cls, info: dict[str, Array]) -> EpisodeStatisticsAccumulator:
        return cls(
            episode_length=info["length"],
            episode_reward=info["reward"],
            episode_done=info["done"],
        )


class JITSummaryWriter:
    """
    A wrapper around `tensorboardX.SummaryWriter` with a JIT compatible interface.
    """

    summary_writer: SummaryWriter

    def __init__(self, log_dir: str | None = None):
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(
        self,
        tag: str,
        scalar_value: ScalarLike,
        global_step: Int[ArrayLike, ""] | None = None,
        walltime: Float[ArrayLike, ""] | None = None,
    ):
        """
        Add a scalar value to the summary writer.
        """
        scalar_value = eqx.error_if(
            scalar_value,
            jnp.logical_or(jnp.isnan(scalar_value), jnp.isinf(scalar_value)),
            "Scalar value cannot be NaN",
        )
        debug_with_numpy_wrapper(self.summary_writer.add_scalar, thread=True)(
            tag, scalar_value, global_step, walltime
        )

    def add_dict(
        self,
        scalars: dict[str, Scalar],
        prefix: str = "",
        *,
        global_step: Int[ArrayLike, ""] | None = None,
        walltime: Float[ArrayLike, ""] | None = None,
    ) -> None:
        """
        Log a dictionary of **scalar** values.
        """

        if prefix:
            scalars = {f"{prefix}/{k}": v for k, v in scalars.items()}

        for tag, value in scalars.items():
            self.add_scalar(tag, value, global_step=global_step, walltime=walltime)

    def log_episode_stats(
        self,
        episode_stats: EpisodeStatisticsAccumulator,
        *,
        global_step: Int[ArrayLike, ""] | None = None,
        walltime: Float[ArrayLike, ""] | None = None,
    ) -> None:
        """
        Log episode statistics to the summary writer.
        """

        def log_fn(rewards, lengths, dones, global_step, walltime):
            global_step = None if global_step is None else global_step - len(rewards)
            for reward, length, done in zip(
                rewards,
                lengths,
                dones,
            ):
                global_step = None if global_step is None else global_step + 1
                if done:
                    self.summary_writer.add_scalar(
                        "episode/reward",
                        reward,
                        global_step=global_step,
                        walltime=walltime,
                    )
                    self.summary_writer.add_scalar(
                        "episode/length",
                        length,
                        global_step=global_step,
                        walltime=walltime,
                    )

        debug_with_numpy_wrapper(log_fn, thread=True)(
            episode_stats.episode_reward,
            episode_stats.episode_length,
            episode_stats.episode_done,
            global_step,
            walltime,
        )


def superscript_digit(digit: int) -> str:
    return "⁰¹²³⁴⁵⁶⁷⁸⁹"[digit]


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

    def start(self) -> None:
        debug_wrapper(self.progress_bar.start, thread=True)()

    def stop(self) -> None:
        debug_wrapper(self.progress_bar.stop, thread=True)()

    def update(
        self,
        total: Float[ArrayLike, ""] | None = None,
        completed: Float[ArrayLike, ""] | None = None,
        advance: Float[ArrayLike, ""] | None = None,
        description: str | None = None,
        visible: Bool[ArrayLike, ""] | None = None,
        refresh: Bool[ArrayLike, ""] = False,
    ) -> None:
        debug_with_list_wrapper(self.progress_bar.update, thread=True)(
            self.task,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
        )
