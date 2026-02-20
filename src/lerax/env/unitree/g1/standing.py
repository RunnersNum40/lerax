"""G1 standing environment: upright balance without locomotion."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import equinox as eqx
import mujoco
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key
from mujoco import mjx

from lerax.space import Box

from . import randomize
from .base_g1 import NUM_ACTUATED_DOFS, AbstractG1Env, G1EnvState
from .gait import initial_gait_phase

OBSERVATION_SIZE = 99


class G1Standing(AbstractG1Env):
    """Unitree G1 standing: maintain upright balance.

    The agent must keep the robot standing upright at the default pose
    without moving. No gait phase is included in the observation since
    the robot should not be walking.

    Observation (99 dims):
        - Local linear velocity (3)
        - Gyroscope angular velocity (3)
        - Gravity vector in body frame (3)
        - Velocity command [0, 0, 0] (3)
        - Joint angles offset from default (29)
        - Joint velocities (29)
        - Last action (29)

    Action: [-1, 1]^29 scaled to joint position targets around default pose.
    """

    name: ClassVar[str] = "G1Standing"

    action_space: Box
    observation_space: Box

    base_model: mjx.Model
    mujoco_model: mujoco.MjModel
    frame_skip: int
    dt: Float[Array, ""]

    action_scale: Float[Array, ""]
    default_joint_positions: Float[Array, "29"]
    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    nominal_friction_loss: Float[Array, "..."]
    nominal_armature: Float[Array, "..."]
    nominal_body_mass: Float[Array, "..."]
    torso_body_id: int

    friction_range: tuple[float, float] = eqx.field(static=True)
    friction_loss_scale_range: tuple[float, float] = eqx.field(static=True)
    armature_scale_range: tuple[float, float] = eqx.field(static=True)
    mass_scale_range: tuple[float, float] = eqx.field(static=True)
    torso_offset_range: tuple[float, float] = eqx.field(static=True)

    push_enable: bool
    push_interval_range: Float[Array, "2"]
    push_magnitude_range: Float[Array, "2"]

    noise_level: Float[Array, ""]
    noise_scales: dict[str, float] = eqx.field(static=True)

    soft_joint_pos_limit_factor: Float[Array, ""]
    joint_lower_limits: Float[Array, "29"]
    joint_upper_limits: Float[Array, "29"]
    soft_joint_lower_limits: Float[Array, "29"]
    soft_joint_upper_limits: Float[Array, "29"]

    pelvis_imu_site_id: int
    torso_imu_site_id: int

    feet_site_ids: tuple[int, int]
    foot_linvel_sensor_slices: tuple[tuple[int, int], tuple[int, int]]

    reward_orientation_weight: Float[Array, ""]
    reward_joint_vel_weight: Float[Array, ""]
    reward_pose_weight: Float[Array, ""]
    reward_alive_weight: Float[Array, ""]
    reward_termination_weight: Float[Array, ""]
    reward_action_rate_weight: Float[Array, ""]

    def __init__(
        self,
        *,
        xml_file: str | Path = "scene_mjx.xml",
        control_frequency_hz: float = 50.0,
        action_scale: float = 0.5,
        keyframe_name: str = "knees_bent",
        soft_joint_pos_limit_factor: float = 0.95,
        push_enable: bool = True,
        push_interval_range: tuple[float, float] = (5.0, 10.0),
        push_magnitude_range: tuple[float, float] = (0.1, 2.0),
        noise_level: float = 1.0,
        noise_scales: dict[str, float] | None = None,
        friction_range: tuple[float, float] = (0.4, 1.0),
        friction_loss_scale_range: tuple[float, float] = (0.5, 2.0),
        armature_scale_range: tuple[float, float] = (1.0, 1.05),
        mass_scale_range: tuple[float, float] = (0.9, 1.1),
        torso_offset_range: tuple[float, float] = (-1.0, 1.0),
        reward_orientation_weight: float = 2.0,
        reward_joint_vel_weight: float = -0.1,
        reward_pose_weight: float = -0.5,
        reward_alive_weight: float = 1.0,
        reward_termination_weight: float = -100.0,
        reward_action_rate_weight: float = -0.01,
    ):
        self._init_common(
            xml_file=xml_file,
            control_frequency_hz=control_frequency_hz,
            action_scale=action_scale,
            keyframe_name=keyframe_name,
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
            push_enable=push_enable,
            push_interval_range=push_interval_range,
            push_magnitude_range=push_magnitude_range,
            noise_level=noise_level,
            noise_scales=noise_scales,
            friction_range=friction_range,
            friction_loss_scale_range=friction_loss_scale_range,
            armature_scale_range=armature_scale_range,
            mass_scale_range=mass_scale_range,
            torso_offset_range=torso_offset_range,
        )

        self.observation_space = Box(
            low=-jnp.inf, high=jnp.inf, shape=(OBSERVATION_SIZE,)
        )

        self.reward_orientation_weight = jnp.array(reward_orientation_weight)
        self.reward_joint_vel_weight = jnp.array(reward_joint_vel_weight)
        self.reward_pose_weight = jnp.array(reward_pose_weight)
        self.reward_alive_weight = jnp.array(reward_alive_weight)
        self.reward_termination_weight = jnp.array(reward_termination_weight)
        self.reward_action_rate_weight = jnp.array(reward_action_rate_weight)

    def initial(self, *, key: Key[Array, ""]) -> G1EnvState:
        randomize_key, qpos_key, qvel_key = jr.split(key, 3)

        model = randomize.randomize_model(
            self.base_model,
            key=randomize_key,
            nominal_friction_loss=self.nominal_friction_loss,
            nominal_armature=self.nominal_armature,
            nominal_body_mass=self.nominal_body_mass,
            torso_body_id=self.torso_body_id,
            friction_range=self.friction_range,
            friction_loss_scale_range=self.friction_loss_scale_range,
            armature_scale_range=self.armature_scale_range,
            mass_scale_range=self.mass_scale_range,
            torso_offset_range=self.torso_offset_range,
        )

        qpos = self.init_qpos.at[7:].set(
            self.init_qpos[7:]
            * jr.uniform(qpos_key, shape=(NUM_ACTUATED_DOFS,), minval=0.9, maxval=1.1)
        )
        qvel = self.init_qvel.at[:6].set(
            jr.uniform(qvel_key, shape=(6,), minval=-0.1, maxval=0.1)
        )

        data = mjx.make_data(model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=qpos[7:])
        data = mjx.forward(model, data)

        return G1EnvState(
            sim_state=data,
            t=jnp.array(0.0),
            model=model,
            last_action=jnp.zeros(NUM_ACTUATED_DOFS),
            gait_phase=initial_gait_phase(),
            gait_frequency=jnp.array(0.0),
            command=jnp.zeros(3),
            step_count=jnp.array(0.0),
            feet_air_time=jnp.zeros(2),
            last_contact=jnp.zeros(2, dtype=bool),
        )

    def observation(
        self, state: G1EnvState, *, key: Key[Array, ""]
    ) -> Float[Array, "99"]:
        data = state.sim_state
        linvel_key, gyro_key, grav_key, jpos_key, jvel_key = jr.split(key, 5)

        linvel = self._local_linvel(data)
        gyro = self._gyro(data)
        gravity = self._gravity_vector(data)
        joint_angles = self._joint_angles_offset(data)
        joint_vel = self._joint_velocities(data)

        def _add_noise(value: Array, noise_key: Key[Array, ""], scale: float) -> Array:
            noise = (
                (2 * jr.uniform(noise_key, shape=value.shape) - 1)
                * self.noise_level
                * scale
            )
            return value + noise

        linvel = _add_noise(linvel, linvel_key, self.noise_scales.get("linvel", 0.0))
        gyro = _add_noise(gyro, gyro_key, self.noise_scales.get("gyro", 0.0))
        gravity = _add_noise(gravity, grav_key, self.noise_scales.get("gravity", 0.0))
        joint_angles = _add_noise(
            joint_angles, jpos_key, self.noise_scales.get("joint_pos", 0.0)
        )
        joint_vel = _add_noise(
            joint_vel, jvel_key, self.noise_scales.get("joint_vel", 0.0)
        )

        return jnp.concatenate(
            [
                linvel,  # 3
                gyro,  # 3
                gravity,  # 3
                state.command,  # 3
                joint_angles,  # 29
                joint_vel,  # 29
                state.last_action,  # 29
            ]
        )

    def reward(
        self,
        state: G1EnvState,
        action: Float[Array, "29"],
        next_state: G1EnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        data = next_state.sim_state

        torso_gravity = self._torso_gravity_vector(data)
        orientation = (
            1.0 - jnp.sum(jnp.square(torso_gravity[:2]))
        ) * self.reward_orientation_weight

        joint_vel = jnp.sum(jnp.square(data.qvel[6:])) * self.reward_joint_vel_weight
        pose = (
            jnp.sum(jnp.square(data.qpos[7:] - self.default_joint_positions))
            * self.reward_pose_weight
        )

        alive = self.reward_alive_weight
        action_rate = (
            jnp.sum(jnp.square(action - state.last_action))
            * self.reward_action_rate_weight
        )

        done = self._is_fallen(data)
        termination = done.astype(float) * self.reward_termination_weight

        return (
            orientation + joint_vel + pose + alive + action_rate + termination
        ) * self.dt

    def terminal(self, state: G1EnvState, *, key: Key[Array, ""]) -> Bool[Array, ""]:
        data = state.sim_state
        fallen = self._is_fallen(data)
        nan_detected = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        self_contact = self._self_contact_termination(data)
        return fallen | nan_detected | self_contact

    def sample_command(self, *, key: Key[Array, ""]) -> Float[Array, "3"]:
        """Standing always uses zero command."""
        return jnp.zeros(3)

    def _is_fallen(self, data: mjx.Data) -> Bool[Array, ""]:
        torso_gravity = self._torso_gravity_vector(data)
        return torso_gravity[2] < 0.0
