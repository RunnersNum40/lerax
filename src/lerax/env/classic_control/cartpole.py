from __future__ import annotations

from typing import ClassVar

import diffrax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.render import (
    Abstract2DRenderer,
    AbstractRenderer,
    Color,
    PygameRenderer,
    Transform,
)
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
    """
    CartPole environment matching the [Gymnasium Cart Pole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

    Note:
        To achieve identical dynamics to Gymnasium set `solver=diffrax.Euler()`.

    ## Action Space

    The action space is discrete with two actions:

    - 0: Push cart to the left
    - 1: Push cart to the right

    The action applies a fixed magnitude force to the cart in the specified direction for the duration of the time step.

    ## Observation Space

    The observation space is a 4-dimensional continuous space representing the state of the cart and pole:

    | Index | Observation           | Min Value           | Max Value          |
    |-------|-----------------------|---------------------|--------------------|
    | 0     | Cart Position         | -4.8                | 4.8                |
    | 1     | Cart Velocity         | -Inf                | Inf                |
    | 2     | Pole Angle            | -24 deg (-0.418 rad)| 24 deg (0.418 rad) |
    | 3     | Pole Angular Velocity | -Inf                | Inf                |

    These values are double the termination thresholds to allow for some margin.
    These limits can be modified via the `theta_threshold_radians` and `x_threshold` parameters.

    ## Reward

    The reward is 1 for every step taken, including the termination step.

    ## Termination

    The episode terminates when:

    - The pole angle exceeds ±12 degrees from vertical.
    - The cart position exceeds ±2.4 units from the center.

    These values can be modified via the `theta_threshold_radians` and `x_threshold` parameters.

    Args:
        gravity: The gravity constant.
        cart_mass: The mass of the cart.
        pole_mass: The mass of the pole.
        half_length: The half-length of the pole.
        force_mag: The magnitude of the force applied to the cart.
        theta_threshold_radians: The angle threshold for terminating the episode.
        x_threshold: The position threshold for terminating the episode.
        dt: The time step for the simulation.
        solver: The differential equation solver used for simulating the dynamics.
        stepsize_controller: The step size controller for the solver.
    """

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
        if not isinstance(renderer, Abstract2DRenderer):
            raise TypeError("CartPole environment requires a 2D renderer.")

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

    def default_renderer(self) -> Abstract2DRenderer:
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
