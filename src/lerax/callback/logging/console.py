from __future__ import annotations

from typing import Any

import equinox as eqx
import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..progress_bar import SpeedColumn
from .backend import AbstractLoggingBackend


class ConsoleBackend(AbstractLoggingBackend):
    """Logging backend that displays a live metrics table and progress bar.

    Uses Rich's ``Live`` display to show a metrics table that updates in-place
    on each ``log_scalars`` call, with a progress bar rendered below it. The
    table is replaced (not appended) on each update so the display stays
    compact.

    When ``total_timesteps`` is ``None`` the progress bar is omitted and
    metrics are printed as simple key=value lines.

    Args:
        total_timesteps: Total training steps for the progress bar.
            When ``None`` no progress bar or live display is shown.
    """

    _total_timesteps: int | None
    _console: Console
    _display: dict[str, Any] = eqx.field(static=True)

    def __init__(self, total_timesteps: int | None = None) -> None:
        self._total_timesteps = total_timesteps
        self._console = Console()
        self._display = {}

    def _build_renderable(self) -> Group:
        """Build the composite renderable: metrics table above progress bar."""
        parts = []
        if "metrics_table" in self._display:
            parts.append(self._display["metrics_table"])
        if "progress" in self._display:
            parts.append(self._display["progress"])
        return Group(*parts)

    def open(self, name: str) -> None:
        if self._total_timesteps is not None:
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
            self._display["progress"] = progress
            self._display["task_id"] = progress.add_task(
                f"[yellow]{name}", total=self._total_timesteps
            )

            live = Live(
                self._build_renderable(),
                console=self._console,
                refresh_per_second=4,
            )
            self._display["live"] = live
            live.start()
        else:
            self._console.print(f"[bold]Run:[/bold] {name}")

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        if not hparams:
            return
        table = Table(title="Hyperparameters", show_header=False, min_width=40)
        table.add_column("name", style="cyan")
        table.add_column("value", style="green")
        for key, value in sorted(hparams.items()):
            table.add_row(key, str(value))
        self._console.print(table)

    def log_scalars(self, scalars: dict[str, np.ndarray], step: int) -> None:
        live = self._display.get("live")
        if live is not None:
            self._display["progress"].update(self._display["task_id"], completed=step)

            table = Table(
                show_header=True,
                header_style="bold",
                min_width=40,
                title=f"[dim]step {int(step)}[/dim]",
                title_justify="left",
            )
            table.add_column("metric", style="cyan")
            table.add_column("value", style="green", justify="right")
            for key, value in sorted(scalars.items()):
                table.add_row(key, f"{float(value):.4g}")
            self._display["metrics_table"] = table

            live.update(self._build_renderable())
        else:
            parts = [f"[dim]step {int(step)}[/dim]"]
            for key, value in sorted(scalars.items()):
                parts.append(f"[cyan]{key}[/cyan]=[green]{float(value):.4g}[/green]")
            self._console.print("  ".join(parts))

    def log_video(self, tag: str, frames: np.ndarray, step: int, fps: float) -> None:
        pass

    def close(self) -> None:
        live = self._display.get("live")
        if live is not None:
            live.stop()
