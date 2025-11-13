from __future__ import annotations

import equinox as eqx
import optax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key, Scalar, ScalarLike
from rich import progress, text
from tensorboardX import SummaryWriter

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicyState, AbstractStatefulPolicy
from lerax.utils import (
    callback_with_list_wrapper,
    callback_with_numpy_wrapper,
    callback_wrapper,
)


class EpisodeStats(eqx.Module):
    episode_return: Float[Array, ""]
    episode_length: Int[Array, ""]
    episode_done: Bool[Array, ""]

    @classmethod
    def initial(cls) -> EpisodeStats:
        return cls(
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
            jnp.array(False, dtype=bool),
        )

    def next(self, reward: Float[Array, ""], done: Bool[Array, ""]) -> EpisodeStats:
        return EpisodeStats(
            self.episode_return * (1.0 - self.episode_done.astype(float)) + reward,
            self.episode_length * (1 - self.episode_done.astype(int)) + 1,
            done,
        )


class StepCarry[PolicyType: AbstractStatefulPolicy](eqx.Module):
    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    episode_stats: EpisodeStats

    @classmethod
    def initial(cls, env: AbstractEnvLike, policy: PolicyType, key: Key) -> StepCarry:
        env_state = env.initial(key=key)
        policy_state = policy.reset()
        return cls(env_state, policy_state, EpisodeStats.initial())


class IterationCarry[PolicyType: AbstractStatefulPolicy](eqx.Module):
    iteration_count: Int[Array, ""]
    step_carry: StepCarry[PolicyType]
    policy: PolicyType
    opt_state: optax.OptState


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
            jnp.isnan(scalar_value) | jnp.isinf(scalar_value),
            "Scalar value cannot be NaN or Inf.",
        )
        callback_with_numpy_wrapper(self.summary_writer.add_scalar)(
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
        episode_stats: EpisodeStats,
        *,
        first_step: Int[ArrayLike, ""],
    ) -> None:
        def log_finished(returns, lengths, dones, step_offset):
            for i, (_return, length, done) in enumerate(zip(returns, lengths, dones)):
                step = step_offset + i
                if done:
                    self.summary_writer.add_scalar(
                        "episode/return", _return, global_step=step
                    )
                    self.summary_writer.add_scalar(
                        "episode/length", length, global_step=step
                    )

        returns = episode_stats.episode_return
        lengths = episode_stats.episode_length
        dones = episode_stats.episode_done
        if returns.ndim == 1:
            returns = returns[None, :]
            lengths = lengths[None, :]
            dones = dones[None, :]

        num_envs, num_steps = returns.shape

        interleaved_returns = jnp.empty((num_steps * num_envs), dtype=returns.dtype)
        interleaved_lengths = jnp.empty((num_steps * num_envs), dtype=lengths.dtype)
        interleaved_dones = jnp.empty((num_steps * num_envs), dtype=dones.dtype)

        for env_idx in range(num_envs):
            interleaved_returns = interleaved_returns.at[env_idx::num_envs].set(
                returns[env_idx]
            )
            interleaved_lengths = interleaved_lengths.at[env_idx::num_envs].set(
                lengths[env_idx]
            )
            interleaved_dones = interleaved_dones.at[env_idx::num_envs].set(
                dones[env_idx]
            )

        callback_with_numpy_wrapper(log_finished)(
            interleaved_returns, interleaved_lengths, interleaved_dones, first_step
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

    def start(self) -> None:
        callback_wrapper(self.progress_bar.start)()

    def stop(self) -> None:
        callback_wrapper(self.progress_bar.stop)()

    def update(
        self,
        total: Float[ArrayLike, ""] | None = None,
        completed: Float[ArrayLike, ""] | None = None,
        advance: Float[ArrayLike, ""] | None = None,
        description: str | None = None,
        visible: Bool[ArrayLike, ""] | None = None,
        refresh: Bool[ArrayLike, ""] = False,
    ) -> None:
        callback_with_list_wrapper(self.progress_bar.update)(
            self.task,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
        )
