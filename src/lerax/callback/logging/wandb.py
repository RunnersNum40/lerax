from __future__ import annotations

from typing import Any

import equinox as eqx
import numpy as np

import wandb

from .backend import AbstractLoggingBackend


class WandbBackend(AbstractLoggingBackend):
    """
    Logging backend that writes to Weights & Biases.

    ``wandb.init`` is called in ``open`` and ``wandb.finish`` is called in
    ``close``.

    Args:
        project: W&B project name.
        config: Hyperparameter dictionary passed to ``wandb.init``.
    """

    _project: str | None = eqx.field(static=True)
    _config: dict[str, Any] | None = eqx.field(static=True)

    def __init__(
        self,
        project: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._project = project
        self._config = config

    def open(self, name: str) -> None:
        wandb.init(project=self._project, name=name, config=self._config)
        wandb.define_metric("eval/*", step_metric="eval_step")

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        wandb.config.update(hparams)

    def log_scalars(self, scalars: dict[str, np.ndarray], step: int) -> None:
        wandb.log({k: float(v) for k, v in scalars.items()}, step=int(step))

    def log_video(self, tag: str, frames: np.ndarray, step: int, fps: float) -> None:
        # wandb expects (T, C, H, W); input is (T, H, W, C).
        video = np.transpose(frames, (0, 3, 1, 2))
        wandb.log(
            {"eval_step": step, tag: wandb.Video(video, fps=int(fps), format="mp4")}
        )

    def close(self) -> None:
        wandb.finish()
