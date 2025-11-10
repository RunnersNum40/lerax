from __future__ import annotations

from typing import ClassVar

import diffrax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.render import AbstractRenderer, Color, PygameRenderer, Transform
from lerax.space import Box, Discrete

from .base_classic_control import (
    AbstractClassicControlEnv,
    AbstractClassicControlEnvState,
)


class CartPoleState(AbstractClassicControlEnvState):
    y: Float[Array, "4"]
    t: Float[Array, ""]


class CartPole(
    AbstractClassicControlEnv[CartPoleState, Int[Array, ""], Float[Array, "4"]]
):
    name: ClassVar[str] = "CartPole"

    action_space: Discrete
    observation_space: Box

    gravity: Float[Array, ""]
    cart_mass: Float[Array, ""]
    pole_mass: Float[Array, ""]
    total_mass: Float[Array, ""]
    length: Float[Array, ""]
    polemass_length: Float[Array, ""]
    force_mag: Float[Array, ""]
    theta_threshold_radians: Float[Array, ""]
    x_threshold: Float[Array, ""]

    dt: Float[Array, ""]
    solver: diffrax.AbstractSolver
    dt0: Float[Array, ""] | None
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(
        self,
        *,
        gravity: Float[ArrayLike, ""] = 9.8,
        cart_mass: Float[ArrayLike, ""] = 1.0,
        pole_mass: Float[ArrayLike, ""] = 0.1,
        half_length: Float[ArrayLike, ""] = 0.5,
        force_mag: Float[ArrayLike, ""] = 10.0,
        theta_threshold_radians: Float[ArrayLike, ""] = 12 * 2 * jnp.pi / 360,
        x_threshold: Float[ArrayLike, ""] = 2.4,
        dt: Float[ArrayLike, ""] = 0.02,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    ):
        self.gravity = jnp.array(gravity)
        self.cart_mass = jnp.array(cart_mass)
        self.pole_mass = jnp.array(pole_mass)
        self.total_mass = self.pole_mass + self.cart_mass
        self.length = jnp.array(half_length)
        self.polemass_length = self.pole_mass * self.length
        self.force_mag = jnp.array(force_mag)

        self.dt = jnp.array(dt)
        self.solver = solver or diffrax.Tsit5()
        is_adaptive = isinstance(self.solver, diffrax.AbstractAdaptiveSolver)
        self.dt0 = None if is_adaptive else self.dt
        if stepsize_controller is None:
            if is_adaptive:
                self.stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
            else:
                self.stepsize_controller = diffrax.ConstantStepSize()
        else:
            self.stepsize_controller = stepsize_controller

        self.theta_threshold_radians = jnp.array(theta_threshold_radians)
        self.x_threshold = jnp.array(x_threshold)

        self.action_space = Discrete(2)
        high = jnp.array(
            [
                self.x_threshold * 2,
                jnp.inf,
                self.theta_threshold_radians * 2,
                jnp.inf,
            ],
        )
        self.observation_space = Box(-high, high)

    def initial(self, *, key: Key) -> CartPoleState:
        return CartPoleState(
            y=jr.uniform(key, (4,), minval=-0.05, maxval=0.05), t=jnp.array(0.0)
        )

    def dynamics(
        self, t: Float[Array, ""], y: Float[Array, "4"], action: Int[Array, ""]
    ) -> Float[Array, "4"]:
        _, x_dot, theta, theta_dot = y
        force = (action * 2 - 1) * self.force_mag

        temp = (
            force + self.polemass_length * theta_dot**2 * jnp.sin(theta)
        ) / self.total_mass
        theta_dd = (self.gravity * jnp.sin(theta) - jnp.cos(theta) * temp) / (
            self.length
            * (4.0 / 3.0 - self.pole_mass * (jnp.cos(theta) ** 2) / self.total_mass)
        )
        x_dd = temp - self.polemass_length * theta_dd * jnp.cos(theta) / self.total_mass

        return jnp.array([x_dot, x_dd, theta_dot, theta_dd])

    def clip(self, y: Float[Array, "4"]) -> Float[Array, "4"]:
        return y

    def observation(self, state: CartPoleState, *, key: Key) -> Float[Array, "4"]:
        return state.y

    def reward(
        self,
        state: CartPoleState,
        action: Int[Array, ""],
        next_state: CartPoleState,
        *,
        key: Key,
    ) -> Float[Array, ""]:
        return jnp.array(1.0)

    def terminal(self, state: CartPoleState, *, key: Key) -> Bool[Array, ""]:
        x, theta = state.y[0], state.y[2]
        within_x = (x >= -self.x_threshold) & (x <= self.x_threshold)
        within_theta = (theta >= -self.theta_threshold_radians) & (
            theta <= self.theta_threshold_radians
        )
        return ~(within_x & within_theta)

    def render(self, state: CartPoleState, renderer: AbstractRenderer):
        x, theta = state.y[0], state.y[2]

        renderer.clear()

        # Ground
        renderer.draw_line(
            start=jnp.array((-10.0, 0.0)),
            end=jnp.array((10.0, 0.0)),
            color=Color(0.0, 0.0, 0.0),
            width=0.01,
        )
        # Cart
        cart_w, cart_h = 0.3, 0.15
        cart_col = Color(0.0, 0.0, 0.0)
        renderer.draw_rect(jnp.array((x, 0.0)), w=cart_w, h=cart_h, color=cart_col)
        # Pole
        pole_start = jnp.asarray((x, cart_h / 4))
        pole_end = pole_start + self.length * jnp.asarray(
            [jnp.sin(theta), jnp.cos(theta)]
        )
        pole_col = Color(0.8, 0.6, 0.4)
        renderer.draw_line(pole_start, pole_end, color=pole_col, width=0.05)
        # Pole Hinge
        hinge_r = 0.025
        hinge_col = Color(0.5, 0.5, 0.5)
        renderer.draw_circle(pole_start, radius=hinge_r, color=hinge_col)

        renderer.draw()

    def default_renderer(self) -> AbstractRenderer:
        width, height = 800, 450
        transform = Transform(
            scale=200.0,
            offset=jnp.array([width / 2, height * 0.1]),
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
