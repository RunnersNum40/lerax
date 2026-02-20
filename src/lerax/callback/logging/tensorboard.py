from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
from tensorboardX import SummaryWriter

from .backend import AbstractLoggingBackend


class TensorBoardBackend(AbstractLoggingBackend):
    """
    Logging backend that writes to TensorBoard via ``tensorboardX``.

    The log directory is determined by ``log_dir / name`` where *name* is
    provided by ``open``.

    Args:
        log_dir: Base directory for TensorBoard event files.
    """

    _log_dir: Path = eqx.field(static=True)
    _writer: Any = eqx.field(static=True)

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self._log_dir = Path(log_dir)
        self._writer = None

    def open(self, name: str) -> None:
        writer = SummaryWriter(log_dir=(self._log_dir / name).as_posix())
        object.__setattr__(self, "_writer", writer)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(hparams.items()))
        self._writer.add_text(
            "hparams", f"| key | value |\n|---|---|\n{rows}", global_step=0
        )

    def log_scalars(self, scalars: dict[str, np.ndarray], step: int) -> None:
        for tag, value in scalars.items():
            self._writer.add_scalar(tag, value, global_step=step)

    def log_video(self, tag: str, frames: np.ndarray, step: int, fps: float) -> None:
        # TensorBoard expects (N, T, C, H, W); input is (T, H, W, C).
        video = np.transpose(frames, (0, 3, 1, 2))[np.newaxis, ...]
        self._writer.add_video(tag, video, step, fps=fps)

    def close(self) -> None:
        self._writer.close()
