from __future__ import annotations

import contextlib
import warnings
from typing import Any

from jax import numpy as jnp
from jaxtyping import ArrayLike, Float

from .base_renderer import WHITE, Abstract2DRenderer, Color, Transform

pygame: Any = None
gfxdraw: Any = None


def _load_pygame() -> tuple[Any, Any]:
    """
    Import pygame and gfxdraw lazily.

    Raises a clear ImportError if the optional dependency is missing.
    """
    global pygame, gfxdraw

    if pygame is not None and gfxdraw is not None:
        return pygame, gfxdraw

    try:
        with contextlib.redirect_stdout(None):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                import pygame as _pygame
                from pygame import gfxdraw as _gfxdraw
    except ImportError as exc:
        raise ImportError(
            "PygameRenderer requires the optional rendering dependencies. "
            "Install them with: pip install lerax[render]"
        ) from exc

    pygame = _pygame
    gfxdraw = _gfxdraw
    return pygame, gfxdraw


class PygameRenderer(Abstract2DRenderer):
    """
    PyGame renderer implementation.

    Attributes:
        transform: Transform from world coordinates to screen pixels.
        width: The width of the rendering window in pixels.
        height: The height of the rendering window in pixels.
        screen: The PyGame surface representing the rendering window.
        background_color: The background color of the rendering window.

    Args:
        width: The width of the rendering window in pixels.
        height: The height of the rendering window in pixels.
        background_color: The background color of the rendering window.
        transform: Transform from world coordinates to screen pixels.
    """

    transform: Transform

    width: int
    height: int

    screen: Any
    background_color: Color

    def __init__(
        self,
        width: int,
        height: int,
        background_color: Color = WHITE,
        transform: Transform | None = None,
    ) -> None:
        pg, _ = _load_pygame()

        self.width = width
        self.height = height
        self.background_color = background_color

        if transform is None:
            self.transform = Transform(
                width=width,
                height=height,
                scale=jnp.array(1.0),
                offset=jnp.array([0.0, 0.0]),
            )
        else:
            self.transform = transform

        pg.init()
        self.screen = pg.display.set_mode((self.width, self.height))

    def open(self):
        pass

    def close(self):
        pg, _ = _load_pygame()
        pg.display.quit()
        pg.quit()

    def draw(self):
        pg, _ = _load_pygame()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return
        pg.display.flip()

    def clear(self):
        self.screen.fill(self._pg_color(self.background_color))

    @staticmethod
    def _pg_color(c: Color) -> Any:
        pg, _ = _load_pygame()
        r, g, b = c.to_rgb255()
        return pg.Color(r, g, b)

    def _to_px(self, point: Float[ArrayLike, "2"]) -> tuple[int, int]:
        return self.transform.world_to_px(
            jnp.asarray(point)
        ).tolist()  # pyright: ignore

    def _scale_x(self, length: Float[ArrayLike, ""]) -> int:
        return int(self.transform.scale_length(length))  # pyright: ignore

    def draw_circle(
        self, center: Float[ArrayLike, "2"], radius: Float[ArrayLike, ""], color: Color
    ):
        _, _gfx = _load_pygame()
        _gfx.aacircle(
            self.screen,
            *self._to_px(center),
            self._scale_x(radius),
            self._pg_color(color),
        )
        _gfx.filled_circle(
            self.screen,
            *self._to_px(center),
            self._scale_x(radius),
            self._pg_color(color),
        )

    def draw_line(
        self,
        start: Float[ArrayLike, "2"],
        end: Float[ArrayLike, "2"],
        color: Color,
        width: Float[ArrayLike, ""] = 1,
    ):
        start = jnp.asarray(start)
        end = jnp.asarray(end)

        # Uses rectangle to work around lack of width in aaline
        norm = jnp.array([start[1] - end[1], end[0] - start[0]])  # pyright: ignore
        norm = norm / jnp.linalg.norm(norm) * (width / 2)

        p1 = start + norm
        p2 = start - norm
        p3 = end - norm
        p4 = end + norm

        self.draw_polygon(jnp.asarray([p1, p2, p3, p4]), color)

    def draw_rect(
        self,
        center: Float[ArrayLike, "2"],
        w: Float[ArrayLike, ""],
        h: Float[ArrayLike, ""],
        color: Color,
    ):
        _, _gfx = _load_pygame()
        pg = pygame

        top_left = self._to_px(
            jnp.asarray(center) - jnp.array([w / 2, -h / 2])  # pyright: ignore
        )
        width_height = (self._scale_x(w), self._scale_x(h))

        rect = pg.Rect(top_left, width_height)

        _gfx.box(self.screen, rect, self._pg_color(color))

    def draw_polygon(self, points: Float[ArrayLike, "num 2"], color: Color):
        _, _gfx = _load_pygame()
        pts = [self._to_px(point) for point in points]  # pyright: ignore
        if len(pts) >= 3:
            _gfx.aapolygon(self.screen, pts, self._pg_color(color))
            _gfx.filled_polygon(self.screen, pts, self._pg_color(color))
        else:
            raise ValueError("Need at least 3 points to draw a polygon.")

    def draw_text(
        self,
        center: Float[ArrayLike, "2"],
        text: str,
        color: Color,
        size: Float[ArrayLike, ""] = 12,
    ):
        pg, _ = _load_pygame()
        if not pg.font.get_init():
            pg.font.init()
        font = pg.font.SysFont(None, int(size))  # pyright: ignore
        surf = font.render(text, True, self._pg_color(color))
        px, py = self._to_px(center)
        self.screen.blit(surf, (px, py))

    def draw_polyline(
        self,
        points: Float[ArrayLike, "num 2"],
        color: Color,
    ):
        pg, _ = _load_pygame()
        if len(points) >= 2:  # pyright: ignore
            pg.draw.aalines(
                self.screen,
                self._pg_color(color),
                False,
                [self._to_px(p) for p in points],  # pyright: ignore
            )
        else:
            raise ValueError("Need at least 2 points to draw a polyline.")

    def draw_ellipse(
        self,
        center: Float[ArrayLike, "2"],
        w: Float[ArrayLike, ""],
        h: Float[ArrayLike, ""],
        color: Color,
    ):
        _, _gfx = _load_pygame()
        px, py = self._to_px(center)
        rx = max(1, int(self._scale_x(w) / 2))
        ry = max(1, int(self._scale_x(h) / 2))

        _gfx.aaellipse(self.screen, px, py, rx, ry, color=self._pg_color(color))

    def as_array(self) -> Float[ArrayLike, "height width 3"]:
        pg, _ = _load_pygame()
        arr = pg.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()
        return arr
