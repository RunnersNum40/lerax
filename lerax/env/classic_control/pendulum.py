from __future__ import annotations

from typing import ClassVar

import diffrax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from lerax.render import AbstractRenderer, Color, PygameRenderer, Transform
from lerax.space import Box

from .base_classic_control import (
    AbstractClassicControlEnv,
    AbstractClassicControlEnvState,
)


class PendulumState(AbstractClassicControlEnvState):
    y: Float[Array, "2"]
    t: Float[Array, ""]


class Pendulum(
    AbstractClassicControlEnv[PendulumState, Float[Array, ""], Float[Array, "3"]]
):
    name: ClassVar[str] = "Pendulum"

    action_space: Box
    observation_space: Box

    max_speed: Float[Array, ""]
    max_torque: Float[Array, ""]
    dt: Float[Array, ""]
    g: Float[Array, ""]
    m: Float[Array, ""]
    l: Float[Array, ""]

    solver: diffrax.AbstractSolver
    dt0: Float[Array, ""] | None
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(
        self,
        *,
        max_speed: Float[ArrayLike, ""] = 8.0,
        max_torque: Float[ArrayLike, ""] = 2.0,
        g: Float[ArrayLike, ""] = 9.8,
        m: Float[ArrayLike, ""] = 1.0,
        l: Float[ArrayLike, ""] = 1.0,
        dt: Float[ArrayLike, ""] = 0.05,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    ):
        self.max_speed = jnp.array(max_speed)
        self.max_torque = jnp.array(max_torque)
        self.dt = jnp.array(dt)
        self.g = jnp.array(g)
        self.m = jnp.array(m)
        self.l = jnp.array(l)

        self.solver = solver or diffrax.Tsit5()
        is_adaptive = isinstance(self.solver, diffrax.AbstractAdaptiveSolver)
        self.dt0 = None if is_adaptive else self.dt
        if stepsize_controller is None:
            self.stepsize_controller = (
                diffrax.PIDController(rtol=1e-5, atol=1e-5)
                if is_adaptive
                else diffrax.ConstantStepSize()
            )
        else:
            self.stepsize_controller = stepsize_controller

        self.action_space = Box(-self.max_torque, self.max_torque)
        high = jnp.array([1.0, 1.0, self.max_speed])
        self.observation_space = Box(-high, high)

    def initial(self, *, key: Key) -> PendulumState:
        high = jnp.array([jnp.pi, 1.0])
        state = jr.uniform(key, shape=(2,), minval=-high, maxval=high)
        return PendulumState(y=state, t=jnp.array(0.0))

    def dynamics(
        self, t: Float[Array, ""], y: Float[Array, "2"], action: Float[Array, ""]
    ) -> Float[Array, "2"]:
        theta, theta_dot = y
        u = jnp.clip(action, -self.max_torque, self.max_torque)
        theta_dd = (
            3.0 * self.g / (2.0 * self.l) * jnp.sin(theta)
            + 3.0 / (self.m * self.l**2) * u
        )
        return jnp.array([theta_dot, theta_dd])

    def clip(self, y: Float[Array, "2"]) -> Float[Array, "2"]:
        theta, theta_dot = y
        theta = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        theta_dot = jnp.clip(theta_dot, -self.max_speed, self.max_speed)
        return jnp.array([theta, theta_dot])

    def observation(self, state: PendulumState, *, key: Key) -> Float[Array, "3"]:
        theta, theta_dot = state.y
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])

    def reward(
        self,
        state: PendulumState,
        action: Float[Array, ""],
        next_state: PendulumState,
        *,
        key: Key,
    ) -> Float[Array, ""]:
        theta, theta_dot = next_state.y
        u = jnp.clip(action, -self.max_torque, self.max_torque)
        cost = theta**2 + 0.1 * theta_dot**2 + 0.001 * (u**2)
        return -cost

    def terminal(self, state: PendulumState, *, key: Key) -> Bool[Array, ""]:
        return jnp.array(False)

    def render(self, state: PendulumState, renderer: AbstractRenderer):
        theta = state.y[0]

        renderer.clear()

        rod_w = 0.2
        origin = jnp.array([0.0, 0.0])
        end = origin + self.l * jnp.array([jnp.sin(theta), jnp.cos(theta)])
        rod_color = Color(0.8, 0.3, 0.3)
        axle_color = Color(0.0, 0.0, 0.0)

        renderer.draw_circle(origin, radius=rod_w / 2, color=rod_color)
        renderer.draw_line(origin, end, color=rod_color, width=rod_w)
        renderer.draw_circle(end, radius=rod_w / 2, color=rod_color)
        renderer.draw_circle(origin, radius=0.05, color=axle_color)

        renderer.draw()

    def default_renderer(self) -> AbstractRenderer:
        width, height = 500, 500
        transform = Transform(
            scale=200.0,
            offset=jnp.array([width / 2, height / 2]),
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
