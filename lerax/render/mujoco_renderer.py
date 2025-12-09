from __future__ import annotations

import glfw
import mujoco
import numpy as np
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float
from mujoco import mjx

from .base_renderer import Abstract3DRenderer


class HeadlessMujocoRenderer:
    """
    MuJoCo renderer for headless (off-screen) rendering.
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

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

    def render(self, data: mjx.Data):
        """
        Render the current frame.

        Args:
            data: The MuJoCo simulation data to render.
        """
        for i in range(3):
            self.camera.lookat[i] = jnp.median(data.geom_xpos[:, i])

        mujoco.mjv_updateScene(
            self.model,
            mjx.get_data(self.model, data),  # pyright: ignore
            self.visual_options,
            self.pert,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        mujoco.mjr_render(self.viewport, self.scene, self.context)

    def draw(self):
        """
        Read the rendered frame into arrays.

        Returns:
            rgb_array: The RGB image array of shape (height, width, 3).
            depth_array: The depth image array of shape (height, width).
        """
        rgb_array = np.zeros((self.height, self.width, 3), dtype=np.uint8).flatten()
        depth_array = np.zeros((self.height, self.width), dtype=np.float32).flatten()

        mujoco.mjr_readPixels(rgb_array, depth_array, self.viewport, self.context)

        rgb_array = rgb_array.reshape((self.height, self.width, 3))[::-1, :]
        depth_array = depth_array.reshape((self.height, self.width))[::-1, :]

        return rgb_array, depth_array


class WindowMujocoRenderer:
    """
    On-screen MuJoCo renderer using GLFW.
    """

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
        """
        Render the current frame.

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

        mujoco.mjv_updateScene(
            self.model,
            mjx.get_data(self.model, data),  # pyright: ignore
            self.visual_options,
            self.pert,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        mujoco.mjr_render(self.viewport, self.scene, self.context)

    def draw(self):
        """
        Draw the rendered frame to the window.
        """
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        """
        Close the rendering window.
        """
        glfw.destroy_window(self.window)
        glfw.terminate()


class MujocoRenderer(Abstract3DRenderer):
    """
    MuJoCo renderer that opens a window for on-screen rendering.

    Attributes:
        renderer: The underlying WindowMujocoRenderer instance.

    Args:
        model: The MuJoCo model to render.
        width: The width of the rendering window in pixels.
        height: The height of the rendering window in pixels.
        max_geom: The maximum number of geometries to render.
        visual_options: A dictionary of visual options for rendering.
        name: The title of the rendering window.
    """

    renderer: WindowMujocoRenderer

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int | None = None,
        height: int | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
        name: str = "MuJoCo",
    ):
        self.renderer = WindowMujocoRenderer(
            model,
            width,
            height,
            max_geom,
            visual_options,
            name,
        )

    def open(self):
        pass

    def close(self):
        self.renderer.close()

    def render(self, data: mjx.Data):
        """
        Update the renderer with the current simulation data.

        Args:
            data: The MuJoCo simulation data to render.
        """
        self.renderer.render(data)

    def draw(self):
        """
        Update the rendering window with the latest rendered frame.
        """
        self.renderer.draw()

    def as_array(self) -> Float[ArrayLike, "H W 3"]:
        raise NotImplementedError(
            "As-array rendering is not supported in windowed mode."
        )
