from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from .backend import AbstractLoggingBackend

_console = Console()


class ConsoleBackend(AbstractLoggingBackend):
    """
    Logging backend that prints metrics to the console using Rich.

    Hyperparameters are printed once as a table at training start.
    Scalars are printed as a compact key=value list at each logged step.
    Video logging is not supported.
    """

    def open(self, name: str) -> None:
        _console.print(f"[bold]Run:[/bold] {name}")

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        if not hparams:
            return
        table = Table(title="Hyperparameters", show_header=False, min_width=40)
        table.add_column("name", style="cyan")
        table.add_column("value", style="green")
        for key, value in sorted(hparams.items()):
            table.add_row(key, str(value))
        _console.print(table)

    def log_scalars(self, scalars: dict[str, np.ndarray], step: int) -> None:
        parts = [f"[dim]step {int(step)}[/dim]"]
        for key, value in sorted(scalars.items()):
            parts.append(f"[cyan]{key}[/cyan]=[green]{float(value):.4g}[/green]")
        _console.print("  ".join(parts))

    def log_video(self, tag: str, frames: np.ndarray, step: int, fps: float) -> None:
        pass

    def close(self) -> None:
        pass
