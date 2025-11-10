from __future__ import annotations

# Disable pygame greeting message and pkg_resources warnings
import contextlib
import warnings

with contextlib.redirect_stdout(None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        import pygame
        from pygame import gfxdraw

from jax import numpy as jnp
from jaxtyping import ArrayLike, Float

from .base_renderer import WHITE, AbstractRenderer, Color, Transform


class PygameRenderer(AbstractRenderer):
    """
    PyGame renderer implementation.
    """

    transform: Transform

    width: int
    height: int

    screen: pygame.Surface
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

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

    def is_open(self) -> bool:
        return True

    def open(self):
        pass

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def draw(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        pygame.display.flip()

    def clear(self):
        self.screen.fill(self._pg_color(self.background_color))

    @staticmethod
    def _pg_color(c: Color) -> pygame.Color:
        r, g, b = c.to_rgb255()
        return pygame.Color(r, g, b)

    def _to_px(self, point: Float[ArrayLike, "2"]) -> tuple[int, int]:
        return self.transform.world_to_px(
            jnp.asarray(point)
        ).tolist()  # pyright: ignore

    def _scale_x(self, length: Float[ArrayLike, ""]) -> int:
        return int(self.transform.scale_length(length))  # pyright: ignore

    def draw_circle(
        self, center: Float[ArrayLike, "2"], radius: Float[ArrayLike, ""], color: Color
    ):
        gfxdraw.aacircle(
            self.screen,
            *self._to_px(center),
            self._scale_x(radius),
            self._pg_color(color),
        )
        gfxdraw.filled_circle(
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
        top_left = self._to_px(
            jnp.asarray(center) - jnp.array([w / 2, -h / 2])  # pyright: ignore
        )
        width_hight = (self._scale_x(w), self._scale_x(h))

        rect = pygame.Rect(top_left, width_hight)

        gfxdraw.box(self.screen, rect, self._pg_color(color))

    def draw_polygon(self, points: Float[ArrayLike, "num 2"], color: Color):
        pts = [self._to_px(point) for point in points]  # pyright: ignore
        if len(pts) >= 3:
            gfxdraw.aapolygon(self.screen, pts, self._pg_color(color))
            gfxdraw.filled_polygon(self.screen, pts, self._pg_color(color))
        else:
            raise ValueError("Need at least 3 points to draw a polygon.")

    def draw_text(
        self,
        center: Float[ArrayLike, "2"],
        text: str,
        color: Color,
        size: Float[ArrayLike, ""] = 12,
    ):
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont(None, int(size))  # pyright: ignore
        surf = font.render(text, True, self._pg_color(color))
        px, py = self._to_px(center)
        self.screen.blit(surf, (px, py))

    def draw_polyline(
        self,
        points: Float[ArrayLike, "num 2"],
        color: Color,
    ):
        if len(points) >= 2:  # pyright: ignore
            pygame.draw.aalines(
                self.screen,
                points=[self._to_px(p) for p in points],  # pyright: ignore
                color=self._pg_color(color),
                closed=False,
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

        gfxdraw.aaellipse(self.screen, px, py, rx, ry, color=self._pg_color(color))

    def as_array(self) -> Float[ArrayLike, "height width 3"]:
        arr = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()
        return arr
