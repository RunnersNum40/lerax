from __future__ import annotations

from abc import abstractmethod

import glfw
import imageio.v2 as imageio
import mujoco
import numpy as np
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float
from mujoco import mjx

from .base_renderer import Abstract3DRenderer


class HeadlessMujocoRenderer:
    """MuJoCo renderer for headless (off-screen) rendering.

    Uses ``mujoco.GLContext`` (EGL/OSMesa) to create a GPU-accelerated
    OpenGL context without requiring a display server or GLFW.
    """

    model: mujoco.MjModel

    width: int
    height: int
    max_geom: int

    viewport: mujoco.MjrRect
    scene: mujoco.MjvScene
    camera: mujoco.MjvCamera
    visual_options: mujoco.MjvOption
    pert: mujoco.MjvPerturb

    markers: list[dict]
    overlays: dict[int, list[str]]
    context: mujoco.MjrContext

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 800,
        height: int = 600,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
    ):
        self.width, self.height = width, height

        self._gl_context = mujoco.GLContext(width, height)
        self._gl_context.make_current()

        self.model = model
        self.markers = []
        self.overlays = {}
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        self.scene = mujoco.MjvScene(self.model, max_geom)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        self.camera.distance = self.model.stat.extent
        self.visual_options = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        for flag, value in visual_options.items():
            self.visual_options.flags[flag] = value

        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

    def render(self, data: mjx.Data):
        """Render the current frame to the offscreen buffer.

        Args:
            data: The MuJoCo simulation data to render.
        """
        for i in range(3):
            self.camera.lookat[i] = jnp.median(data.geom_xpos[:, i])

        mj_data = mjx.get_data(self.model, data)
        if isinstance(mj_data, list):
            mj_data = mj_data[0]
        assert isinstance(mj_data, mujoco.MjData)

        mujoco.mjv_updateScene(
            self.model,
            mj_data,
            self.visual_options,
            self.pert,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        mujoco.mjr_render(self.viewport, self.scene, self.context)

    def read_pixels(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the rendered frame from the offscreen buffer.

        Returns:
            A tuple of (rgb_array, depth_array) where rgb_array has shape
            (height, width, 3) and depth_array has shape (height, width).
        """
        rgb_array = np.zeros((self.height, self.width, 3), dtype=np.uint8).flatten()
        depth_array = np.zeros((self.height, self.width), dtype=np.float32).flatten()

        mujoco.mjr_readPixels(rgb_array, depth_array, self.viewport, self.context)

        rgb_array = rgb_array.reshape((self.height, self.width, 3))[::-1, :]
        depth_array = depth_array.reshape((self.height, self.width))[::-1, :]

        return rgb_array, depth_array

    def close(self):
        """Free the GL context."""
        self._gl_context.free()


class WindowMujocoRenderer:
    """On-screen MuJoCo renderer using GLFW."""

    model: mujoco.MjModel

    window: glfw._GLFWwindow
    width: int
    height: int
    max_geom: int

    viewport: mujoco.MjrRect
    scene: mujoco.MjvScene
    camera: mujoco.MjvCamera
    visual_options: mujoco.MjvOption
    pert: mujoco.MjvPerturb

    markers: list[dict]
    overlays: dict[int, list[str]]
    context: mujoco.MjrContext

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int | None = None,
        height: int | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
        name: str = "MuJoCo",
    ):
        glfw.init()

        monitor_width, monitor_height = glfw.get_video_mode(
            glfw.get_primary_monitor()
        ).size

        width = monitor_width // 2 if width is None else width
        height = monitor_height // 2 if height is None else height

        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(width, height, name, None, None)
        self.width, self.height = glfw.get_framebuffer_size(self.window)

        self.model = model
        self.markers = []
        self.overlays = {}
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        self.scene = mujoco.MjvScene(self.model, max_geom)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        self.camera.distance = self.model.stat.extent
        self.visual_options = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        for flag, value in visual_options.items():
            self.visual_options.flags[flag] = value

        glfw.make_context_current(self.window)

        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

        glfw.swap_interval(1)

    def render(self, data: mjx.Data):
        """Render the current frame.

        Args:
            data: The MuJoCo simulation data to render.
        """
        if glfw.window_should_close(self.window):
            glfw.destroy_window(self.window)
            glfw.terminate()

        for i in range(3):
            self.camera.lookat[i] = jnp.median(data.geom_xpos[:, i])

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
            self.window
        )

        mj_data = mjx.get_data(self.model, data)
        if isinstance(mj_data, list):
            mj_data = mj_data[0]
        assert isinstance(mj_data, mujoco.MjData)

        mujoco.mjv_updateScene(
            self.model,
            mj_data,
            self.visual_options,
            self.pert,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        mujoco.mjr_render(self.viewport, self.scene, self.context)

    def draw(self):
        """Draw the rendered frame to the window."""
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        """Close the rendering window."""
        glfw.destroy_window(self.window)
        glfw.terminate()


class AbstractMujocoRenderer(Abstract3DRenderer):
    """Base class for MuJoCo renderers.

    Extends the generic 3D renderer with the MuJoCo-specific
    ``render(data)`` method for feeding simulation state.
    """

    @abstractmethod
    def render(self, data: mjx.Data):
        """Update the scene with simulation data.

        Args:
            data: MuJoCo simulation data to render.
        """


class MujocoRenderer(AbstractMujocoRenderer):
    """MuJoCo renderer supporting both windowed and headless modes.

    In windowed mode (default), opens an on-screen GLFW window for
    interactive viewing. In headless mode, renders offscreen and supports
    ``as_array()`` for capturing frames as numpy arrays.

    Args:
        model: The MuJoCo model to render.
        headless: If True, use offscreen rendering instead of a window.
        width: Width in pixels. Defaults to half monitor width (windowed)
            or 800 (headless).
        height: Height in pixels. Defaults to half monitor height (windowed)
            or 600 (headless).
        max_geom: Maximum number of geometries to render.
        visual_options: Dictionary of MuJoCo visual option flags.
        name: Window title (ignored in headless mode).
    """

    renderer: WindowMujocoRenderer | HeadlessMujocoRenderer

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        headless: bool = False,
        width: int | None = None,
        height: int | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
        name: str = "MuJoCo",
    ):
        if headless:
            self.renderer = HeadlessMujocoRenderer(
                model,
                width=width or 800,
                height=height or 600,
                max_geom=max_geom,
                visual_options=visual_options,
            )
        else:
            self.renderer = WindowMujocoRenderer(
                model,
                width=width,
                height=height,
                max_geom=max_geom,
                visual_options=visual_options,
                name=name,
            )

    def open(self):
        pass

    def close(self):
        self.renderer.close()

    def render(self, data: mjx.Data):
        """Update the renderer with the current simulation data.

        Args:
            data: The MuJoCo simulation data to render.
        """
        self.renderer.render(data)

    def draw(self):
        """Display the latest rendered frame.

        In windowed mode, swaps the display buffers. In headless mode,
        this is a no-op.
        """
        if isinstance(self.renderer, WindowMujocoRenderer):
            self.renderer.draw()

    def as_array(self) -> Float[ArrayLike, "H W 3"]:
        """Return the current frame as an RGB array.

        Only supported in headless mode.

        Returns:
            RGB image array of shape (height, width, 3).

        Raises:
            NotImplementedError: If called in windowed mode.
        """
        if isinstance(self.renderer, HeadlessMujocoRenderer):
            rgb, _depth = self.renderer.read_pixels()
            return rgb
        raise NotImplementedError(
            "as_array() is not supported in windowed mode. "
            "Use headless=True for array-based rendering."
        )


class MujocoVideoRenderer(AbstractMujocoRenderer):
    """MuJoCo renderer that records frames to a video file.

    Wraps a headless ``MujocoRenderer`` and captures each frame when
    ``draw()`` is called. When ``close()`` is called, all accumulated
    frames are written to the output video file.

    Example::

        renderer = MujocoVideoRenderer(env.mujoco_model, "rollout.mp4", fps=50.0)
        env.render_stacked(states, renderer=renderer)

    Args:
        model: The MuJoCo model to render.
        output_path: Path to the output video file.
        fps: Frames per second for the output video.
        width: Width of the rendered frames in pixels.
        height: Height of the rendered frames in pixels.
        max_geom: Maximum number of geometries to render.
        visual_options: Dictionary of MuJoCo visual option flags.
    """

    inner: MujocoRenderer
    output_path: str
    fps: float
    frames: list

    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: str,
        *,
        fps: float = 50.0,
        width: int = 800,
        height: int = 600,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
    ):
        self.inner = MujocoRenderer(
            model,
            headless=True,
            width=width,
            height=height,
            max_geom=max_geom,
            visual_options=visual_options,
        )
        self.output_path = output_path
        self.fps = fps
        self.frames = []

    def open(self):
        self.inner.open()

    def close(self):
        if self.frames:
            writer = imageio.get_writer(
                self.output_path, fps=self.fps, macro_block_size=1
            )
            try:
                for frame in self.frames:
                    writer.append_data(frame)
            finally:
                writer.close()
        self.inner.close()

    def render(self, data: mjx.Data):
        """Update the scene with simulation data.

        Args:
            data: MuJoCo simulation data to render.
        """
        self.inner.render(data)

    def draw(self):
        """Capture the current frame for video recording."""
        self.inner.draw()
        frame = self.inner.as_array()
        self.frames.append(frame)

    def as_array(self) -> Float[ArrayLike, "H W 3"]:
        """Return the current frame as an RGB array.

        Returns:
            RGB image array of shape (height, width, 3).
        """
        return self.inner.as_array()
