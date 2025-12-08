from abc import abstractmethod

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int


class Color(eqx.Module):
    """RGB in [0,1]. Helpers to convert to pygame-friendly formats."""

    r: Float[Array, ""]
    g: Float[Array, ""]
    b: Float[Array, ""]

    def __init__(
        self, r: Float[ArrayLike, ""], g: Float[ArrayLike, ""], b: Float[ArrayLike, ""]
    ):
        self.r = jnp.asarray(r)
        self.g = jnp.asarray(g)
        self.b = jnp.asarray(b)

    def to_tuple(
        self,
    ) -> tuple[Float[ArrayLike, ""], Float[ArrayLike, ""], Float[ArrayLike, ""]]:
        return (self.r, self.g, self.b)

    def to_rgb255(self) -> tuple[int, int, int]:
        r = int(float(jnp.clip(self.r, 0.0, 1.0)) * 255.0)
        g = int(float(jnp.clip(self.g, 0.0, 1.0)) * 255.0)
        b = int(float(jnp.clip(self.b, 0.0, 1.0)) * 255.0)
        return (r, g, b)

    def to_hex(self) -> str:
        r, g, b = self.to_rgb255()
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        r = int(hex_str[1:3], 16) / 255.0
        g = int(hex_str[3:5], 16) / 255.0
        b = int(hex_str[5:7], 16) / 255.0
        return cls(r, g, b)

    @classmethod
    def from_rgb255(cls, r: int, g: int, b: int) -> "Color":
        return cls(r / 255.0, g / 255.0, b / 255.0)

    @classmethod
    def from_tuple(
        cls,
        rgb: tuple[Float[ArrayLike, ""], Float[ArrayLike, ""], Float[ArrayLike, ""]],
    ) -> "Color":
        return cls(*rgb)


WHITE = Color(1.0, 1.0, 1.0)
BLACK = Color(0.0, 0.0, 0.0)
GRAY = Color(0.78, 0.78, 0.78)
RED = Color(0.86, 0.24, 0.24)
GREEN = Color(0.24, 0.71, 0.29)
BLUE = Color(0.26, 0.53, 0.96)


class Transform(eqx.Module):
    """
    Affine mapping from world coords to screen pixels.

    Has a constant scale for all axis to avoid distortion.

    Attributes:
        width: Width of the screen in pixels.
        height: Height of the screen in pixels.
        scale: Scale factor from world units to pixels.
        offset: Offset to add after scaling.
        y_up: If True, the y-axis points up in world space.

    Args:
        width: Width of the screen in pixels.
        height: Height of the screen in pixels.
        scale: Scale factor from world units to pixels.
        offset: Offset to add after scaling.
        y_up: If True, the y-axis points up in world space.
    """

    width: Int[Array, ""]
    height: Int[Array, ""]

    scale: Float[Array, ""]
    offset: Float[Array, "2"]

    y_up: bool = True

    def __init__(
        self,
        width: Int[ArrayLike, ""],
        height: Int[ArrayLike, ""],
        scale: Float[ArrayLike, ""],
        offset: Float[ArrayLike, "2"],
        y_up: bool = True,
    ):
        self.width = jnp.asarray(width)
        self.height = jnp.asarray(height)
        self.scale = jnp.asarray(scale)
        self.offset = jnp.asarray(offset)
        self.y_up = y_up

    def world_to_px(self, point: Float[ArrayLike, "2"]) -> Int[Array, "2"]:
        """
        Return pixel coordinates corresponding to a point in world space.

        Args:
            point: A point in world space.

        Returns:
            The corresponding pixel coordinates.
        """
        pixel = jnp.asarray(point) * self.scale + self.offset

        if self.y_up:
            pixel = pixel.at[1].set(self.height - pixel[1])

        return pixel.astype(int)

    def scale_length(self, length: Float[ArrayLike, ""]) -> Float[Array, ""]:
        """
        Scale a length in world space to pixel space.

        Args:
            length: A length in world space.

        Returns:
            The corresponding length in pixel space.
        """
        return jnp.asarray(length * self.scale)


class AbstractRenderer(eqx.Module):
    """
    Renderer interface.

    Note:
        Renderers are not necessarily thread-safe or safe for JIT compilation.

    Attributes:
        transform: Transform from world space to pixel space.
    """

    transform: eqx.AbstractVar[Transform]

    @abstractmethod
    def is_open(self) -> bool: ...
    @abstractmethod
    def open(self): ...
    @abstractmethod
    def close(self): ...
    @abstractmethod
    def draw(self): ...
    @abstractmethod
    def clear(self): ...

    @abstractmethod
    def draw_circle(
        self, center: Float[ArrayLike, "2"], radius: Float[ArrayLike, ""], color: Color
    ): ...
    @abstractmethod
    def draw_line(
        self,
        start: Float[ArrayLike, "2"],
        end: Float[ArrayLike, "2"],
        color: Color,
        width: Float[ArrayLike, ""] = 1,
    ): ...
    @abstractmethod
    def draw_rect(
        self,
        center: Float[ArrayLike, "2"],
        w: Float[ArrayLike, ""],
        h: Float[ArrayLike, ""],
        color: Color,
    ): ...
    @abstractmethod
    def draw_polygon(self, points: Float[ArrayLike, "num 2"], color: Color): ...
    @abstractmethod
    def draw_text(
        self,
        center: Float[ArrayLike, "2"],
        text: str,
        color: Color,
        size: Float[ArrayLike, ""] = 12,
    ): ...
    @abstractmethod
    def draw_polyline(
        self,
        points: Float[ArrayLike, "num 2"],
        color: Color,
    ): ...
    @abstractmethod
    def draw_ellipse(
        self,
        center: Float[ArrayLike, "2"],
        w: Float[ArrayLike, ""],
        h: Float[ArrayLike, ""],
        color: Color,
    ): ...
    @abstractmethod
    def as_array(self) -> Float[ArrayLike, "H W 3"]: ...
