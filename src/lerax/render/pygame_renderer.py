from __future__ import annotations

import contextlib
from typing import Any

# Suppress PyGame's welcome message on import
with contextlib.redirect_stdout(None):
    import pygame as pg

from jax import numpy as jnp
from jaxtyping import ArrayLike, Float
from pygame import gfxdraw as gfx

from .base_renderer import WHITE, Abstract2DRenderer, Color, Transform


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
        pg.display.quit()
        pg.quit()

    def draw(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return
        pg.display.flip()

    def clear(self):
        self.screen.fill(self._pg_color(self.background_color))

    @staticmethod
    def _pg_color(c: Color) -> Any:
        r, g, b = c.to_rgb255()
        return pg.Color(r, g, b)

    def _to_px(self, point: Float[ArrayLike, "2"]) -> tuple[int, int]:
        return tuple(self.transform.world_to_px(jnp.asarray(point)).tolist())

    def _scale_x(self, length: Float[ArrayLike, ""]) -> int:
        return int(self.transform.scale_length(length))

    def draw_circle(
        self, center: Float[ArrayLike, "2"], radius: Float[ArrayLike, ""], color: Color
    ):
        gfx.aacircle(
            self.screen,
            *self._to_px(center),
            self._scale_x(radius),
            self._pg_color(color),
        )
        gfx.filled_circle(
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
        norm = jnp.array([start[1] - end[1], end[0] - start[0]])
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
        w = jnp.asarray(w)
        h = jnp.asarray(h)

        top_left = self._to_px(jnp.asarray(center) - jnp.array([w / 2, -h / 2]))
        width_height = (self._scale_x(w), self._scale_x(h))

        rect = pg.Rect(top_left, width_height)

        gfx.box(self.screen, rect, self._pg_color(color))

    def draw_polygon(self, points: Float[ArrayLike, "num 2"], color: Color):
        pts = [self._to_px(point) for point in jnp.asarray(points)]
        if len(pts) >= 3:
            gfx.aapolygon(self.screen, pts, self._pg_color(color))
            gfx.filled_polygon(self.screen, pts, self._pg_color(color))
        else:
            raise ValueError("Need at least 3 points to draw a polygon.")

    def draw_text(
        self,
        center: Float[ArrayLike, "2"],
        text: str,
        color: Color,
        size: Float[ArrayLike, ""] = 12,
    ):
        if not pg.font.get_init():
            pg.font.init()
        font = pg.font.SysFont(None, int(jnp.asarray(size)))
        surf = font.render(text, True, self._pg_color(color))
        px, py = self._to_px(center)
        self.screen.blit(surf, (px, py))

    def draw_polyline(
        self,
        points: Float[ArrayLike, "num 2"],
        color: Color,
    ):
        points = jnp.asarray(points)
        if len(points) >= 2:
            pg.draw.aalines(
                self.screen,
                self._pg_color(color),
                False,
                [self._to_px(p) for p in points],
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
        px, py = self._to_px(center)
        rx = max(1, int(self._scale_x(w) / 2))
        ry = max(1, int(self._scale_x(h) / 2))

        gfx.aaellipse(self.screen, px, py, rx, ry, color=self._pg_color(color))

    def as_array(self) -> Float[ArrayLike, "height width 3"]:
        arr = pg.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()
        return arr
