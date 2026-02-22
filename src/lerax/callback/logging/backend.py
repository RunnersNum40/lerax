from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import numpy as np


class AbstractLoggingBackend(eqx.Module):
    """Abstract base class for logging backends.

    Implementations receive already-converted Python/numpy values; the
    ``LoggingCallback`` handles the JIT-to-numpy boundary.

    The lifecycle is ``open`` -> ``log_*`` -> ``close``. The ``open`` method
    is called by ``LoggingCallback.__init__`` with the run name at
    construction time. ``close`` is called by ``LoggingCallback.close()``.
    """

    @abstractmethod
    def open(self, name: str) -> None:
        """
        Initialise the backend with the given run name.

        Called exactly once before any ``log_*`` method. Backends should create
        their writer, run handle, or output directory here.

        Args:
            name: Human-readable run name.
        """

    def on_training_start(self, total_timesteps: int, total_iterations: int) -> None:
        """
        Called when training starts.

        Override this method to initialize resources that depend on training
        duration, such as progress bars. The default implementation does nothing.

        Args:
            total_timesteps: Total number of environment steps for this training run.
            total_iterations: Total number of training iterations for this run.
        """

    @abstractmethod
    def log_hparams(self, hparams: dict[str, Any]) -> None:
        """
        Log hyperparameters once at the start of training.

        Args:
            hparams: Hyperparameter names and their Python-scalar values.
        """

    @abstractmethod
    def log_scalars(self, scalars: dict[str, np.ndarray], step: int) -> None:
        """
        Log a dictionary of scalar values.

        Args:
            scalars: Scalar values keyed by metric name.
            step: Current training step.
        """

    @abstractmethod
    def log_video(self, tag: str, frames: np.ndarray, step: int, fps: float) -> None:
        """
        Log a video clip.

        Args:
            tag: Metric name for the video.
            frames: Video frames as a uint8 array of shape ``(T, H, W, C)``.
            step: Current training step.
            fps: Playback frames per second.
        """

    @abstractmethod
    def close(self) -> None:
        """Flush pending data and release any held resources."""
