from __future__ import annotations

from typing import ClassVar, Literal

import diffrax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from lerax.render import AbstractRenderer, Color, PygameRenderer, Transform
from lerax.space import Box

from .base_env import AbstractEnv, AbstractEnvState


class ContinuousMountainCarState(AbstractEnvState):
    y: Float[Array, "2"]


class ContinuousMountainCar(
    AbstractEnv[ContinuousMountainCarState, Float[Array, ""], Float[Array, "2"]]
):
    name: ClassVar[str] = "ContinuousMountainCar"

    action_space: Box
    observation_space: Box

    min_action: float
    max_action: float
    min_position: float
    max_position: float
    max_speed: float
    goal_position: float
    goal_velocity: float

    power: float

    low: Float[Array, "2"]
    high: Float[Array, "2"]

    dt: float
    solver: diffrax.AbstractSolver
    dt0: float | None
    stepsize_controller: diffrax.AbstractStepSizeController

    renderer: AbstractRenderer | None

    def __init__(
        self,
        goal_velocity: float = 0,
        *,
        renderer: AbstractRenderer | Literal["auto"] | None = None,
        solver: diffrax.AbstractSolver | None = None,
        tau: float = 1.0,
    ):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low = jnp.array([self.min_position, -self.max_speed])
        self.high = jnp.array([self.max_position, self.max_speed])

        self.action_space = Box(self.min_action, self.max_action)
        self.observation_space = Box(self.low, self.high)

        if renderer == "auto":
            self.renderer = self.default_renderer()
        else:
            self.renderer = renderer

        self.dt = float(tau)
        self.solver = solver or diffrax.Tsit5()
        is_adaptive = isinstance(self.solver, diffrax.AbstractAdaptiveSolver)
        self.dt0 = None if is_adaptive else self.dt
        self.stepsize_controller = (
            diffrax.PIDController(rtol=1e-5, atol=1e-5)
            if is_adaptive
            else diffrax.ConstantStepSize()
        )

    def initial(self, *, key: Key) -> ContinuousMountainCarState:
        return ContinuousMountainCarState(
            jnp.asarray([jr.uniform(key, minval=-0.6, maxval=-0.4), 0.0])
        )

    def transition(
        self, state: ContinuousMountainCarState, action: Float[Array, ""], *, key: Key
    ) -> ContinuousMountainCarState:

        def rhs(t, y, args):
            x, x_d = y
            force = jnp.clip(action, self.min_action, self.max_action)

            x_dd = self.power * force - 0.0025 * jnp.cos(3.0 * x)
            return jnp.array([x_d, x_dd])

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            self.solver,
            t0=0.0,
            t1=self.dt,
            dt0=self.dt0,
            y0=state.y,
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(t1=True),
        )
        assert sol.ys is not None
        x, v = sol.ys[0]

        v = jnp.clip(v, -self.max_speed, self.max_speed)
        x = jnp.clip(x, self.min_position, self.max_position)
        return ContinuousMountainCarState(jnp.asarray([x, v]))

    def observation(
        self, state: ContinuousMountainCarState, *, key: Key
    ) -> Float[Array, "2"]:
        return state.y

    def reward(
        self,
        state: ContinuousMountainCarState,
        action: Float[Array, ""],
        next_state: ContinuousMountainCarState,
        *,
        key: Key,
    ) -> Float[Array, ""]:
        return (
            100.0 * self.terminal(state, key=key).astype(float)
            - 0.1 * jnp.clip(action, self.min_action, self.max_action) ** 2
        )

    def terminal(
        self, state: ContinuousMountainCarState, *, key: Key
    ) -> Bool[Array, ""]:
        x, v = state.y
        return (x >= self.goal_position) & (v >= self.goal_velocity)

    def truncate(self, state: ContinuousMountainCarState) -> Bool[Array, ""]:
        return jnp.array(False)

    def state_info(self, state: ContinuousMountainCarState) -> dict:
        return {}

    def transition_info(
        self,
        state: ContinuousMountainCarState,
        action: Float[Array, ""],
        next_state: ContinuousMountainCarState,
    ) -> dict:
        return {}

    def render(self, state: ContinuousMountainCarState):
        x = state.y[0]

        assert self.renderer is not None, "Renderer is not initialized."
        self.renderer.clear()

        # Track
        xs = jnp.linspace(self.min_position - 0.5, self.max_position + 0.1, 64)
        ys = jnp.sin(3 * xs) * 0.45
        track_points = jnp.stack([xs, ys], axis=1)
        track_color = Color(0.0, 0.0, 0.0)
        self.renderer.draw_polyline(track_points, color=track_color)

        # Flag
        flag_h = 0.2
        flag_start = jnp.array(
            [self.goal_position, jnp.sin(3 * self.goal_position) * 0.45]
        )
        flag_end = flag_start + jnp.array([0.0, flag_h])

        flag_pole_color = Color(0.0, 0.0, 0.0)
        flag_color = Color(0.86, 0.24, 0.24)
        self.renderer.draw_line(
            flag_start, flag_end, color=flag_pole_color, width=0.005
        )
        flag_points = jnp.array(
            [
                flag_end,
                flag_end + jnp.array([0.1, -0.03]),
                flag_end + jnp.array([0.0, -0.06]),
            ]
        )
        self.renderer.draw_polygon(flag_points, color=flag_color)

        # Car
        car_h, car_w, wheel_r, clearance = 0.04, 0.1, 0.02, 0.025
        angle = jnp.arctan2(jnp.cos(3 * x) * 0.45 * 3, 1.0)
        rot = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )
        ## Body
        car_col = Color(0.0, 0.0, 0.0)
        clearance_vec = rot @ jnp.array([0.0, clearance + car_h / 2])
        car_center = jnp.array([x, jnp.sin(3 * x) * 0.45]) + clearance_vec

        car_corners = jnp.array(
            [
                [-car_w / 2, -car_h / 2],
                [car_w / 2, -car_h / 2],
                [car_w / 2, car_h / 2],
                [-car_w / 2, car_h / 2],
            ]
        )
        car_corners = (rot @ car_corners.T).T + car_center
        self.renderer.draw_polygon(car_corners, color=car_col)
        ## Wheels
        wheel_col = Color(0.3, 0.3, 0.3)
        wheel_clearance = rot @ jnp.array([0.0, wheel_r])
        wheel_centers = jnp.array(
            [
                [-car_w / 4, 0.0],
                [car_w / 4, 0.0],
            ]
        )
        wheel_centers = (
            (rot @ wheel_centers.T).T
            + jnp.array([x, jnp.sin(3 * x) * 0.45])
            + wheel_clearance
        )
        self.renderer.draw_circle(wheel_centers[0], radius=wheel_r, color=wheel_col)
        self.renderer.draw_circle(wheel_centers[1], radius=wheel_r, color=wheel_col)

        self.renderer.render()

    def default_renderer(self) -> AbstractRenderer:
        width, height = 800, 450
        transform = Transform(
            scale=350.0,
            offset=jnp.array([width * 0.7, height * 0.4]),
            width=width,
            height=height,
            y_up=True,
        )
        return PygameRenderer(
            width=width,
            height=height,
            background_color=Color(1.0, 1.0, 1.0),
            transform=transform,
        )

    def close(self):
        pass
