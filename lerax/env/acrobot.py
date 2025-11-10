from __future__ import annotations

from typing import ClassVar, Literal

import diffrax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Int, Key

from lerax.render import AbstractRenderer, Color, PygameRenderer, Transform
from lerax.space import Box, Discrete

from .base_env import AbstractEnv, AbstractEnvState


class AcrobotState(AbstractEnvState):
    y: Float[Array, "4"]


class Acrobot(AbstractEnv[AcrobotState, Int[Array, ""], Float[Array, "4"]]):
    name: ClassVar[str] = "Acrobot"

    action_space: Discrete
    observation_space: Box

    gravity: float
    link_length_1: float
    link_length_2: float
    link_mass_1: float
    link_mass_2: float
    link_com_pos_1: float
    link_com_pos_2: float
    link_moi: float
    max_vel_1: float
    max_vel_2: float
    torque_max_noise: float
    torques: Float[Array, "3"]

    dt: float
    solver: diffrax.AbstractSolver
    dt0: float | None
    stepsize_controller: diffrax.AbstractStepSizeController

    renderer: AbstractRenderer | None

    def __init__(
        self,
        *,
        renderer: AbstractRenderer | Literal["auto"] | None = None,
        solver: diffrax.AbstractSolver | None = None,
        dt: float = 0.2,
    ):
        self.gravity = 9.8
        self.link_length_1 = 1.0
        self.link_length_2 = 1.0
        self.link_mass_1 = 1.0
        self.link_mass_2 = 1.0
        self.link_com_pos_1 = 0.5
        self.link_com_pos_2 = 0.5
        self.link_moi = 1.0
        self.max_vel_1 = 4 * jnp.pi
        self.max_vel_2 = 9 * jnp.pi
        self.torque_max_noise = 0.0
        self.torques = jnp.array([-1.0, 0.0, 1.0])

        self.dt = dt
        self.solver = solver or diffrax.Tsit5()
        is_adaptive = isinstance(self.solver, diffrax.AbstractAdaptiveSolver)
        self.dt0 = None if is_adaptive else self.dt
        self.stepsize_controller = (
            diffrax.PIDController(rtol=1e-5, atol=1e-5)
            if is_adaptive
            else diffrax.ConstantStepSize()
        )

        self.action_space = Discrete(3)
        state_high = jnp.array([1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2])
        low = -state_high
        self.observation_space = Box(low=low, high=state_high)

        if renderer == "auto":
            self.renderer = self.default_renderer()
        else:
            self.renderer = renderer

    def initial(self, *, key: Key) -> AcrobotState:
        return AcrobotState(y=jr.uniform(key, shape=(4,), minval=-0.1, maxval=0.1))

    def transition(
        self, state: AcrobotState, action: Int[Array, ""], *, key: Key
    ) -> AcrobotState:
        def acrobot_ode(t, y, args):
            theta1, theta2, theta1_d, theta2_d, a = y

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
            ) / (
                self.link_mass_2 * self.link_com_pos_2**2 + self.link_moi - d2**2 / d1
            )
            theta1_dd = -(d2 * theta2_dd + phi1) / d1
            return jnp.array([theta1_d, theta2_d, theta1_dd, theta2_dd, 0.0])

        torque = self.torques[action] + jr.uniform(
            key, (), minval=-self.torque_max_noise, maxval=self.torque_max_noise
        )
        y0 = jnp.concatenate([state.y, jnp.array([torque])])

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(acrobot_ode),
            self.solver,
            t0=0.0,
            t1=self.dt,
            dt0=self.dt0,
            stepsize_controller=self.stepsize_controller,
            y0=y0,
            saveat=diffrax.SaveAt(t1=True),
        )

        assert sol.ys is not None
        joint_angle_1, joint_angle_2, joint_vel_1, joint_vel_2 = sol.ys[0][:4]
        joint_angle_1 = (joint_angle_1 + jnp.pi) % (2 * jnp.pi) - jnp.pi
        joint_angle_2 = (joint_angle_2 + jnp.pi) % (2 * jnp.pi) - jnp.pi
        joint_vel_1 = jnp.clip(joint_vel_1, -self.max_vel_1, self.max_vel_1)
        joint_vel_2 = jnp.clip(joint_vel_2, -self.max_vel_2, self.max_vel_2)

        return AcrobotState(
            y=jnp.array([joint_angle_1, joint_angle_2, joint_vel_1, joint_vel_2])
        )

    def observation(self, state: AcrobotState, *, key: Key) -> Float[Array, "4"]:
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
        key: Key,
    ) -> Float[Array, ""]:
        joint_angle_1, joint_angle_2 = next_state.y[0], next_state.y[1]
        done_angle = (
            -jnp.cos(joint_angle_1) - jnp.cos(joint_angle_1 + joint_angle_2) > 1.0
        )
        return done_angle.astype(float) - 1.0

    def terminal(self, state: AcrobotState, *, key: Key) -> Bool[Array, ""]:
        joint_angle_1, joint_angle_2 = state.y[0], state.y[1]
        done_angle = (
            -jnp.cos(joint_angle_1) - jnp.cos(joint_angle_1 + joint_angle_2) > 1.0
        )
        return done_angle

    def truncate(self, state: AcrobotState) -> Bool[Array, ""]:
        return jnp.array(False)

    def state_info(self, state: AcrobotState) -> dict:
        return {}

    def transition_info(
        self,
        state: AcrobotState,
        action: Int[Array, ""],
        next_state: AcrobotState,
    ) -> dict:
        return {}

    def render(self, state: AcrobotState) -> None:
        raise NotImplementedError("Render method is not implemented for Acrobot.")

    def default_renderer(self) -> AbstractRenderer:
        width, height = 800, 450
        transform = Transform(
            width=width,
            height=height,
            scale=100.0,
            offset=jnp.array([width / 2, height / 2 + 100]),
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
