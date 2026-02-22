from __future__ import annotations

import time
from typing import Any

import asciichartpy
import equinox as eqx
import numpy as np
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from ..progress_bar import SpeedColumn
from .backend import AbstractLoggingBackend


class _DisplayState(eqx.Module):
    """Base class for display state."""


class _InteractiveDisplayState(_DisplayState):
    """State for interactive terminal display."""

    progress: Progress
    task_id: TaskID
    live: Live
    metrics_table: Table | None = None
    reward_history: list[float] = eqx.field(default_factory=list)


class _NonInteractiveDisplayState(_DisplayState):
    """State for non-interactive terminal display."""

    total_timesteps: int
    start_time: float


class _InteractiveDisplay(eqx.Module):
    """Rich live display for interactive terminals."""

    _max_history: int = 50
    _chart_height: int = 16
    _y_axis_width: int = 10

    def _build_reward_panel(
        self, state: _InteractiveDisplayState, console: Console
    ) -> Panel:
        if not state.reward_history:
            return Panel(
                Text("Waiting for data...", style="dim"),
                title="[bold]Reward[/bold]",
                border_style="dim",
                expand=True,
            )

        panel_width = console.width - 45
        chart_width = max(20, panel_width - self._y_axis_width - 2)

        data = state.reward_history
        if len(data) < chart_width:
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, chart_width)
            data = np.interp(x_new, x_old, data).tolist()

        cfg = {"height": self._chart_height, "format": "{:8.1f} "}
        chart_str = asciichartpy.plot(data, cfg)
        chart_text = Text(chart_str, style="green")

        content = Group(chart_text)
        return Panel(
            content,
            title="[bold]Reward[/bold]",
            border_style="white",
            expand=True,
        )

    def _build_renderable(
        self, state: _InteractiveDisplayState, console: Console
    ) -> Group:
        parts = []

        if state.reward_history or state.metrics_table is not None:
            if state.metrics_table is not None and state.reward_history:
                grid = Table.grid(expand=True, padding=0)
                grid.add_column()
                grid.add_column(ratio=1)
                grid.add_row(
                    state.metrics_table, self._build_reward_panel(state, console)
                )
                parts.append(grid)
            elif state.metrics_table is not None:
                parts.append(state.metrics_table)
            elif state.reward_history:
                parts.append(self._build_reward_panel(state, console))

        parts.append(state.progress)
        return Group(*parts)

    def start(
        self, name: str, total_timesteps: int, console: Console
    ) -> _InteractiveDisplayState:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(finished_text="[green]✔"),
            MofNCompleteColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("]"),
            SpeedColumn(),
        )
        task_id = progress.add_task(f"[yellow]{name}", total=total_timesteps)

        live = Live(Group(progress), console=console, refresh_per_second=4)
        state = _InteractiveDisplayState(
            progress=progress, task_id=task_id, live=live, reward_history=[]
        )
        live.start()
        return state

    def log_scalars(
        self,
        state: _InteractiveDisplayState,
        scalars: dict[str, np.ndarray],
        step: int,
        console: Console,
    ) -> _InteractiveDisplayState:
        state.progress.update(state.task_id, completed=step)

        table = Table(show_header=True, header_style="bold", min_width=40, box=ROUNDED)
        table.add_column("metric", style="cyan")
        table.add_column("value", style="green", justify="right")
        for key, value in sorted(scalars.items()):
            table.add_row(key, f"{float(value):.4g}")

        object.__setattr__(state, "metrics_table", table)

        if "episode/return" in scalars:
            reward = float(scalars["episode/return"])
            history = list(state.reward_history)
            history.append(reward)
            if len(history) > self._max_history:
                history = history[-self._max_history :]
            object.__setattr__(state, "reward_history", history)

        state.live.update(self._build_renderable(state, console))
        return state

    def stop(self, state: _InteractiveDisplayState) -> None:
        state.live.stop()


class _NonInteractiveDisplay(eqx.Module):
    """Simple text display for non-interactive terminals."""

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 0:
            return "--"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes, secs = divmod(int(seconds), 60)
            return f"{minutes}m{secs}s"
        else:
            hours, remainder = divmod(int(seconds), 3600)
            minutes, secs = divmod(remainder, 60)
            return f"{hours}h{minutes}m"

    def start(
        self, name: str, total_timesteps: int, console: Console
    ) -> _NonInteractiveDisplayState:
        console.print(f"[bold]Run:[/bold] {name}")
        return _NonInteractiveDisplayState(
            total_timesteps=total_timesteps,
            start_time=time.monotonic(),
        )

    def log_scalars(
        self,
        state: _NonInteractiveDisplayState,
        scalars: dict[str, np.ndarray],
        step: int,
        console: Console,
    ) -> _NonInteractiveDisplayState:
        elapsed = time.monotonic() - state.start_time

        if state.total_timesteps > 0:
            percent = (step / state.total_timesteps) * 100
            if step > 0 and elapsed > 0:
                steps_per_second = step / elapsed
                remaining_steps = state.total_timesteps - step
                eta_seconds = remaining_steps / steps_per_second
                eta_str = self._format_duration(eta_seconds)
            else:
                eta_str = "--"

            elapsed_str = self._format_duration(elapsed)
            parts = [
                f"[dim]step {int(step)}[/dim]",
                f"[dim]{percent:5.1f}%[/dim]",
                f"[dim]elapsed:[/dim] {elapsed_str}",
                f"[dim]ETA:[/dim] {eta_str}",
            ]
        else:
            parts = [f"[dim]step {int(step)}[/dim]"]

        for key, value in sorted(scalars.items()):
            parts.append(f"[cyan]{key}[/cyan]=[green]{float(value):.4g}[/green]")
        console.print("  ".join(parts))
        return state

    def stop(self, state: _NonInteractiveDisplayState) -> None:
        pass


class ConsoleBackend(AbstractLoggingBackend):
    """
    Logging backend that displays a live metrics table and progress bar.

    Uses Rich's ``Live`` display to show a metrics table that updates in-place
    on each ``log_scalars`` call, with a progress bar rendered below it. The
    table is replaced (not appended) on each update so the display stays
    compact.

    Progress bar display is automatically enabled when running in an interactive
    terminal. In non-interactive environments (logs, CI, redirected output),
    progress info (percentage, elapsed time, ETA) is logged with each scalar
    update.
    """

    _console: Console
    _name: str = eqx.field(static=True)
    _interactive_display: _InteractiveDisplay = eqx.field(static=True)
    _non_interactive_display: _NonInteractiveDisplay = eqx.field(static=True)
    _interactive_state: _InteractiveDisplayState | None = eqx.field(static=True)
    _non_interactive_state: _NonInteractiveDisplayState | None = eqx.field(static=True)

    def __init__(self) -> None:
        self._console = Console()
        self._name = ""
        self._interactive_display = _InteractiveDisplay()
        self._interactive_state = None
        self._non_interactive_display = _NonInteractiveDisplay()
        self._non_interactive_state = None

    def open(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def on_training_start(self, total_timesteps: int, total_iterations: int) -> None:
        if self._interactive_state is not None:
            self._interactive_display.stop(self._interactive_state)
            object.__setattr__(self, "_interactive_state", None)

        if total_timesteps > 0 and self._console.is_terminal:
            state = self._interactive_display.start(
                self._name, total_timesteps, self._console
            )
            object.__setattr__(self, "_interactive_state", state)
        elif total_timesteps > 0:
            state = self._non_interactive_display.start(
                self._name, total_timesteps, self._console
            )
            object.__setattr__(self, "_non_interactive_state", state)
        else:
            self._console.print(f"[bold]Run:[/bold] {self._name}")

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        if not hparams:
            return
        table = Table(
            title="[bold]Hyperparameters[/bold]",
            show_header=False,
            min_width=40,
            expand=True,
            box=ROUNDED,
        )
        table.add_column("name", style="cyan")
        table.add_column("value", style="green")
        for key, value in sorted(hparams.items()):
            table.add_row(key, str(value))
        self._console.print(table)

    def log_scalars(self, scalars: dict[str, np.ndarray], step: int) -> None:
        if self._interactive_state is not None:
            state = self._interactive_display.log_scalars(
                self._interactive_state, scalars, step, self._console
            )
            object.__setattr__(self, "_interactive_state", state)
        elif self._non_interactive_state is not None:
            state = self._non_interactive_display.log_scalars(
                self._non_interactive_state, scalars, step, self._console
            )
            object.__setattr__(self, "_non_interactive_state", state)
        else:
            parts = [f"[dim]step {int(step)}[/dim]"]
            for key, value in sorted(scalars.items()):
                parts.append(f"[cyan]{key}[/cyan]=[green]{float(value):.4g}[/green]")
            self._console.print("  ".join(parts))

    def log_video(self, tag: str, frames: np.ndarray, step: int, fps: float) -> None:
        pass

    def close(self) -> None:
        if self._interactive_state is not None:
            self._interactive_display.stop(self._interactive_state)
