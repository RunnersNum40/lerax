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


class AcrobotState(AbstractClassicControlEnvState):
    y: Float[Array, "4"]
    t: Float[Array, ""]


class Acrobot(
    AbstractClassicControlEnv[AcrobotState, Int[Array, ""], Float[Array, "4"]]
):
    """
    Acrobot environment matching the [Gymnasium Acrobot environment](https://gymnasium.farama.org/environments/classic_control/acrobot/).

    Note:
        To achieve identical dynamics to Gymnasium set `solver=diffrax.Euler()`.

    ## Action Space

    The action space is discrete with three actions:

    - 0: Apply -1 torque to the joint between the two links
    - 1: Apply 0 torque
    - 2: Apply +1 torque to the joint between the two links

    The action applies a fixed magnitude torque to the joint for the duration of the time step.

    ## Observation Space

    The observation space is a 6-dimensional continuous space representing the state of the two links:

    | Index | Observation               | Min Value | Max Value |
    |-------|---------------------------|-----------|-----------|
    | 0     | Cosine of Joint Angle 1   | -1.0      | 1.0       |
    | 1     | Sine of Joint Angle 1     | -1.0      | 1.0       |
    | 2     | Cosine of Joint Angle 2   | -1.0      | 1.0       |
    | 3     | Sine of Joint Angle 2     | -1.0      | 1.0       |
    | 4     | Joint Velocity 1          | -4π       | 4π        |
    | 5     | Joint Velocity 2          | -9π       | 9π        |

    ## Reward

    Non-terminal steps yield a reward of -1.0 and the terminal step yields a reward of 0.0.

    ## Termination

    The episode terminates when the tip of the second link reaches a height greater than 1.0 unit above the base.
    This corresponds to the condition: -cos(theta1) - cos(theta1 + theta2) > 1.0

    Args:
        gravity: Gravitational acceleration.
        link_length_1: Length of the first link.
        link_length_2: Length of the second link.
        link_mass_1: Mass of the first link.
        link_mass_2: Mass of the second link.
        link_com_pos_1: Center of mass position of the first link.
        link_com_pos_2: Center of mass position of the second link.
        link_moi: Moment of inertia of the links.
        max_vel_1: Maximum angular velocity for the first joint.
        max_vel_2: Maximum angular velocity for the second joint.
        torque_max_noise: Maximum noise to add to the applied torque.
        torques: Array of possible torques corresponding to each action.
        dt: Time step for integration.
        solver: Diffrax solver to use for integration.
        stepsize_controller: Step size controller for adaptive solvers.
    """

    name: ClassVar[str] = "Acrobot"

    action_space: Discrete
    observation_space: Box

    gravity: Float[Array, ""]
    link_length_1: Float[Array, ""]
    link_length_2: Float[Array, ""]
    link_mass_1: Float[Array, ""]
    link_mass_2: Float[Array, ""]
    link_com_pos_1: Float[Array, ""]
    link_com_pos_2: Float[Array, ""]
    link_moi: Float[Array, ""]
    max_vel_1: Float[Array, ""]
    max_vel_2: Float[Array, ""]
    torque_max_noise: Float[Array, ""]
    torques: Float[Array, "3"]

    dt: Float[Array, ""]
    solver: diffrax.AbstractSolver
    dt0: Float[Array, ""] | None
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(
        self,
        *,
        gravity: Float[ArrayLike, ""] = 9.8,
        link_length_1: Float[ArrayLike, ""] = 1.0,
        link_length_2: Float[ArrayLike, ""] = 1.0,
        link_mass_1: Float[ArrayLike, ""] = 1.0,
        link_mass_2: Float[ArrayLike, ""] = 1.0,
        link_com_pos_1: Float[ArrayLike, ""] = 0.5,
        link_com_pos_2: Float[ArrayLike, ""] = 0.5,
        link_moi: Float[ArrayLike, ""] = 1.0,
        max_vel_1: Float[ArrayLike, ""] = 4 * jnp.pi,
        max_vel_2: Float[ArrayLike, ""] = 9 * jnp.pi,
        torque_max_noise: Float[ArrayLike, ""] = 0.0,
        torques: Float[ArrayLike, "3"] = jnp.array([-1.0, 0.0, 1.0]),
        dt: Float[ArrayLike, ""] = 0.2,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    ):
        self.gravity = jnp.array(gravity)
        self.link_length_1 = jnp.array(link_length_1)
        self.link_length_2 = jnp.array(link_length_2)
        self.link_mass_1 = jnp.array(link_mass_1)
        self.link_mass_2 = jnp.array(link_mass_2)
        self.link_com_pos_1 = jnp.array(link_com_pos_1)
        self.link_com_pos_2 = jnp.array(link_com_pos_2)
        self.link_moi = jnp.array(link_moi)
        self.max_vel_1 = jnp.array(max_vel_1)
        self.max_vel_2 = jnp.array(max_vel_2)
        self.torque_max_noise = jnp.array(torque_max_noise)
        self.torques = jnp.array(torques)

        self.dt = jnp.array(dt)
        self.solver = solver or diffrax.Tsit5()
        self.stepsize_controller = stepsize_controller or diffrax.ConstantStepSize()
        self.dt0 = (
            None
            if isinstance(
                self.stepsize_controller, diffrax.AbstractAdaptiveStepSizeController
            )
            else self.dt
        )

        self.action_space = Discrete(3)
        state_high = jnp.array([1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2])
        low = -state_high
        self.observation_space = Box(low=low, high=state_high)

    def initial(self, *, key: Key[Array, ""]) -> AcrobotState:
        return AcrobotState(
            y=jr.uniform(key, shape=(4,), minval=-0.1, maxval=0.1), t=jnp.array(0.0)
        )

    def dynamics(
        self, t: Float[Array, ""], y: Float[Array, "4"], action: Int[Array, ""]
    ) -> Float[Array, "4"]:
        theta1, theta2, theta1_d, theta2_d = y
        a = self.torques[action]

        d1 = (
            self.link_mass_1 * self.link_com_pos_1**2
            + self.link_mass_2
            * (
                self.link_length_1**2
                + self.link_com_pos_2**2
                + 2 * self.link_length_1 * self.link_com_pos_2 * jnp.cos(theta2)
            )
            + self.link_moi
            + self.link_moi
        )
        d2 = (
            self.link_mass_2
            * (
                self.link_com_pos_2**2
                + self.link_length_1 * self.link_com_pos_2 * jnp.cos(theta2)
            )
            + self.link_moi
        )

        phi2 = (
            self.link_mass_2
            * self.link_com_pos_2
            * self.gravity
            * jnp.cos(theta1 + theta2 - jnp.pi / 2)
        )
        phi1 = (
            -self.link_mass_2
            * self.link_length_1
            * self.link_com_pos_2
            * theta2_d**2
            * jnp.sin(theta2)
            - 2
            * self.link_mass_2
            * self.link_length_1
            * self.link_com_pos_2
            * theta1_d
            * theta2_d
            * jnp.sin(theta2)
            + (
                self.link_mass_1 * self.link_com_pos_1
                + self.link_mass_2 * self.link_length_1
            )
            * self.gravity
            * jnp.cos(theta1 - jnp.pi / 2)
            + phi2
        )

        theta2_dd = (
            a
            + d2 / d1 * phi1
            - self.link_mass_2
            * self.link_length_1
            * self.link_com_pos_2
            * theta1_d**2
            * jnp.sin(theta2)
            - phi2
        ) / (self.link_mass_2 * self.link_com_pos_2**2 + self.link_moi - d2**2 / d1)
        theta1_dd = -(d2 * theta2_dd + phi1) / d1
        return jnp.array([theta1_d, theta2_d, theta1_dd, theta2_dd])

    def clip(self, y: Float[Array, "4"]) -> Float[Array, "4"]:
        joint_angle_1, joint_angle_2, joint_vel_1, joint_vel_2 = y
        joint_angle_1 = (joint_angle_1 + jnp.pi) % (2 * jnp.pi) - jnp.pi
        joint_angle_2 = (joint_angle_2 + jnp.pi) % (2 * jnp.pi) - jnp.pi
        joint_vel_1 = jnp.clip(joint_vel_1, -self.max_vel_1, self.max_vel_1)
        joint_vel_2 = jnp.clip(joint_vel_2, -self.max_vel_2, self.max_vel_2)

        return jnp.array([joint_angle_1, joint_angle_2, joint_vel_1, joint_vel_2])

    def observation(
        self, state: AcrobotState, *, key: Key[Array, ""]
    ) -> Float[Array, "4"]:
        joint_angle_1, joint_angle_2, joint_vel_1, joint_vel_2 = state.y
        return jnp.array(
            [
                jnp.cos(joint_angle_1),
                jnp.sin(joint_angle_1),
                jnp.cos(joint_angle_2),
                jnp.sin(joint_angle_2),
                joint_vel_1,
                joint_vel_2,
            ]
        )

    def reward(
        self,
        state: AcrobotState,
        action: Int[Array, ""],
        next_state: AcrobotState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        joint_angle_1, joint_angle_2 = next_state.y[0], next_state.y[1]
        done_angle = (
            -jnp.cos(joint_angle_1) - jnp.cos(joint_angle_1 + joint_angle_2) > 1.0
        )
        return done_angle.astype(float) - 1.0

    def terminal(self, state: AcrobotState, *, key: Key[Array, ""]) -> Bool[Array, ""]:
        joint_angle_1, joint_angle_2 = state.y[0], state.y[1]
        done_angle = (
            -jnp.cos(joint_angle_1) - jnp.cos(joint_angle_1 + joint_angle_2) > 1.0
        )
        return done_angle

    def render(self, state: AcrobotState, renderer: AbstractRenderer):
        if not isinstance(renderer, Abstract2DRenderer):
            raise TypeError("Acrobot environment requires an Abstract2DRenderer.")

        th1, th2 = state.y[0], state.y[1]

        base = jnp.array([0.0, 0.0])
        p1 = base + jnp.array(
            [self.link_length_1 * jnp.sin(th1), -self.link_length_1 * jnp.cos(th1)]
        )
        p2 = p1 + jnp.array(
            [
                self.link_length_2 * jnp.sin(th1 + th2),
                -self.link_length_2 * jnp.cos(th1 + th2),
            ]
        )

        link_w = 0.2
        joint_r = link_w / 2
        link_color = Color(0.0, 0.8, 0.8)
        joint_color = Color(0.8, 0.8, 0.0)
        goal_color = Color(0.0, 0.0, 0.0)

        renderer.clear()

        y_goal = jnp.array(1.0)
        renderer.draw_line(
            start=jnp.array([-3.0, y_goal]),
            end=jnp.array([3.0, y_goal]),
            color=goal_color,
            width=0.01,
        )

        renderer.draw_line(base, p1, color=link_color, width=link_w)
        renderer.draw_line(p1, p2, color=link_color, width=link_w)

        renderer.draw_circle(base, radius=joint_r, color=joint_color)
        renderer.draw_circle(p1, radius=joint_r, color=joint_color)

        renderer.draw()

    def default_renderer(self) -> Abstract2DRenderer:
        width, height = 800, 600
        transform = Transform(
            width=width,
            height=height,
            scale=140.0,
            offset=jnp.array([width / 2, height / 2]),
            y_up=True,
        )
        return PygameRenderer(
            width=width,
            height=height,
            background_color=Color(1.0, 1.0, 1.0),
            transform=transform,
        )
