from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float

from .base_renderer import AbstractRenderer, Color, Transform


class VideoRenderer(AbstractRenderer):
    """
    Renderer wrapper that records frames to a video file.

    It forwards all drawing operations to the underlying renderer and captures
    the framebuffer on each call to `draw()`. When `close()` is called, the
    accumulated frames are written to `output_path` as a video.

    Note:
        This renderer is not JIT-safe and is intended for use in Python loops
        (e.g. via `env.render_states` / `env.render_stacked`).

    Attributes:
        transform: Transform from world space to pixel space.
        inner: The underlying renderer to which drawing operations are forwarded.
        output_path: Path to the output video file.
        fps: Frames per second for the output video.
        frames: List of captured frames.

    Args:
        inner: The underlying renderer to which drawing operations are forwarded.
        output_path: Path to the output video file.
        fps: Frames per second for the output video.
    """

    transform: Transform

    inner: AbstractRenderer
    output_path: str
    fps: float

    frames: list

    def __init__(
        self,
        inner: AbstractRenderer,
        output_path: str,
        fps: float = 60.0,
    ) -> None:
        self.inner = inner
        self.transform = inner.transform
        self.output_path = output_path
        self.fps = fps
        self.frames = []

    def is_open(self) -> bool:
        return self.inner.is_open()

    def open(self):
        self.inner.open()

    def close(self):
        if self.frames:
            writer = imageio.get_writer(self.output_path, fps=self.fps)
            try:
                for frame in self.frames:
                    arr = jnp.asarray(frame)
                    arr = jnp.clip(arr, 0, 255).astype(jnp.uint8)
                    writer.append_data(np.array(arr))
            finally:
                writer.close()
        self.inner.close()

    def draw(self):
        self.inner.draw()
        frame = self.inner.as_array()
        self.frames.append(frame)

    def clear(self):
        self.inner.clear()

    def draw_circle(
        self,
        center: Float[ArrayLike, "2"],
        radius: Float[ArrayLike, ""],
        color: Color,
    ):
        self.inner.draw_circle(center, radius, color)

    def draw_line(
        self,
        start: Float[ArrayLike, "2"],
        end: Float[ArrayLike, "2"],
        color: Color,
        width: Float[ArrayLike, ""] = 1,
    ):
        self.inner.draw_line(start, end, color, width)

    def draw_rect(
        self,
        center: Float[ArrayLike, "2"],
        w: Float[ArrayLike, ""],
        h: Float[ArrayLike, ""],
        color: Color,
    ):
        self.inner.draw_rect(center, w, h, color)

    def draw_polygon(
        self,
        points: Float[ArrayLike, "num 2"],
        color: Color,
    ):
        self.inner.draw_polygon(points, color)

    def draw_text(
        self,
        center: Float[ArrayLike, "2"],
        text: str,
        color: Color,
        size: Float[ArrayLike, ""] = 12,
    ):
        self.inner.draw_text(center, text, color, size)

    def draw_polyline(
        self,
        points: Float[ArrayLike, "num 2"],
        color: Color,
    ):
        self.inner.draw_polyline(points, color)

    def draw_ellipse(
        self,
        center: Float[ArrayLike, "2"],
        w: Float[ArrayLike, ""],
        h: Float[ArrayLike, ""],
        color: Color,
    ):
        self.inner.draw_ellipse(center, w, h, color)

    def as_array(self) -> Float[ArrayLike, "H W 3"]:
        return self.inner.as_array()
