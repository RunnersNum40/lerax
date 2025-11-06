from __future__ import annotations

from typing import ClassVar, Literal

import diffrax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Int, Key

from lerax.render import AbstractRenderer, Color, PygameRenderer, Transform
from lerax.space import Box, Discrete

from .base_env import AbstractEnv, AbstractEnvState


class CartPoleState(AbstractEnvState):
    y: Float[Array, "4"]


class CartPole(AbstractEnv[CartPoleState, Int[Array, ""], Float[Array, "4"]]):
    name: ClassVar[str] = "CartPole"

    action_space: Discrete
    observation_space: Box

    gravity: float
    masscart: float
    masspole: float
    total_mass: float
    length: float
    polemass_length: float
    force_mag: float
    tau: float
    theta_threshold_radians: float
    x_threshold: float

    renderer: AbstractRenderer | None

    solver: diffrax.AbstractSolver
    dt0: float | None
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(
        self,
        *,
        renderer: AbstractRenderer | Literal["auto"] | None = None,
        solver: diffrax.AbstractSolver | None = None,
        tau: float = 0.02,
    ):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0

        self.tau = tau
        self.solver = solver or diffrax.Heun()
        is_adaptive = isinstance(self.solver, diffrax.AbstractAdaptiveSolver)
        self.dt0 = None if is_adaptive else self.tau
        self.stepsize_controller = (
            diffrax.PIDController(rtol=1e-5, atol=1e-5)
            if is_adaptive
            else diffrax.ConstantStepSize()
        )

        self.theta_threshold_radians = 12 * 2 * jnp.pi / 360
        self.x_threshold = 2.4

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

        if renderer == "auto":
            self.renderer = self.default_renderer()
        else:
            self.renderer = renderer

    def initial(self, *, key: Key) -> CartPoleState:
        return CartPoleState(jr.uniform(key, (4,), minval=-0.05, maxval=0.05))

    def transition(
        self, state: CartPoleState, action: Int[Array, ""], *, key: Key
    ) -> CartPoleState:
        def rhs(t, y, args):
            _, x_dot, theta, theta_dot = y
            force = (action * 2 - 1) * self.force_mag

            temp = (
                force + self.polemass_length * theta_dot**2 * jnp.sin(theta)
            ) / self.total_mass
            theta_dd = (self.gravity * jnp.sin(theta) - jnp.cos(theta) * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * (jnp.cos(theta) ** 2) / self.total_mass)
            )
            x_dd = (
                temp
                - self.polemass_length * theta_dd * jnp.cos(theta) / self.total_mass
            )

            return jnp.array([x_dot, x_dd, theta_dot, theta_dd])

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            self.solver,
            t0=0.0,
            t1=self.tau,
            dt0=self.dt0,
            stepsize_controller=self.stepsize_controller,
            y0=state.y,
            saveat=diffrax.SaveAt(t1=True),
        )

        assert sol.ys is not None
        return CartPoleState(sol.ys[0])

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

    def truncate(self, state: CartPoleState) -> Bool[Array, ""]:
        return jnp.array(False)

    def state_info(self, state: CartPoleState) -> dict:
        return {}

    def transition_info(
        self, state: CartPoleState, action: Int[Array, ""], next_state: CartPoleState
    ) -> dict:
        return {}

    def render(self, state: CartPoleState):
        x, theta = state.y[0], state.y[2]

        assert self.renderer is not None, "Renderer is not set."
        self.renderer.clear()

        # Ground
        self.renderer.draw_line(
            start=jnp.array((-10.0, 0.0)),
            end=jnp.array((10.0, 0.0)),
            color=Color(0.0, 0.0, 0.0),
            width=0.01,
        )
        # Cart
        cart_w, cart_h = 0.3, 0.15
        cart_col = Color(0.0, 0.0, 0.0)
        self.renderer.draw_rect(jnp.array((x, 0.0)), w=cart_w, h=cart_h, color=cart_col)
        # Pole
        pole_start = jnp.asarray((x, cart_h / 4))
        pole_end = pole_start + self.length * jnp.asarray(
            [jnp.sin(theta), jnp.cos(theta)]
        )
        pole_col = Color(0.8, 0.6, 0.4)
        self.renderer.draw_line(pole_start, pole_end, color=pole_col, width=0.05)
        # Pole Hinge
        hinge_r = 0.025
        hinge_col = Color(0.5, 0.5, 0.5)
        self.renderer.draw_circle(pole_start, radius=hinge_r, color=hinge_col)

        self.renderer.render()

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

    def close(self):
        pass
