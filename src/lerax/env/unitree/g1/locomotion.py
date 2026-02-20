"""G1 locomotion environment: velocity command tracking with gait phase."""

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

from . import gait, randomize
from .base_g1 import NUM_ACTUATED_DOFS, AbstractG1Env, G1EnvState

OBSERVATION_SIZE = 103


class G1Locomotion(AbstractG1Env):
    """Unitree G1 locomotion: track velocity commands with a natural gait.

    The agent controls 29 joint position targets (legs, waist, arms) and must
    track randomly sampled linear and angular velocity commands while
    maintaining a stable bipedal gait. Domain randomization is applied
    per-episode for sim-to-real transfer.

    Observation (103 dims):
        - Local linear velocity (3)
        - Gyroscope angular velocity (3)
        - Gravity vector in body frame (3)
        - Velocity command [vx, vy, yaw_rate] (3)
        - Joint angles offset from default (29)
        - Joint velocities (29)
        - Last action (29)
        - Gait phase [cos_l, sin_l, cos_r, sin_r] (4)

    Action: [-1, 1]^29 scaled to joint position targets around default pose.

    Reward: Weighted sum of velocity tracking, energy penalties, gait phase
    tracking, and pose regularization, multiplied by dt.
    """

    name: ClassVar[str] = "G1Locomotion"

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

    lin_vel_x_range: Float[Array, "2"]
    lin_vel_y_range: Float[Array, "2"]
    ang_vel_yaw_range: Float[Array, "2"]
    gait_frequency_range: Float[Array, "2"]
    zero_command_probability: Float[Array, ""]

    tracking_sigma: Float[Array, ""]
    max_foot_height: Float[Array, ""]

    reward_tracking_lin_vel: Float[Array, ""]
    reward_tracking_ang_vel: Float[Array, ""]
    reward_ang_vel_xy: Float[Array, ""]
    reward_orientation: Float[Array, ""]
    reward_torques: Float[Array, ""]
    reward_action_rate: Float[Array, ""]
    reward_dof_acc: Float[Array, ""]
    reward_feet_slip: Float[Array, ""]
    reward_feet_air_time: Float[Array, ""]
    reward_feet_phase: Float[Array, ""]
    reward_alive: Float[Array, ""]
    reward_termination: Float[Array, ""]
    reward_stand_still: Float[Array, ""]
    reward_collision: Float[Array, ""]
    reward_dof_pos_limits: Float[Array, ""]
    reward_pose: Float[Array, ""]
    reward_joint_deviation_hip: Float[Array, ""]
    reward_joint_deviation_knee: Float[Array, ""]

    pose_weights: Float[Array, "29"]

    hip_indices: tuple[int, ...]
    knee_indices: tuple[int, ...]

    command_resample_steps: int

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
        lin_vel_x_range: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y_range: tuple[float, float] = (-0.5, 0.5),
        ang_vel_yaw_range: tuple[float, float] = (-1.0, 1.0),
        gait_frequency_range: tuple[float, float] = (1.25, 1.5),
        zero_command_probability: float = 0.1,
        tracking_sigma: float = 0.25,
        max_foot_height: float = 0.15,
        reward_weights: dict[str, float] | None = None,
        command_resample_steps: int = 500,
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

        self.lin_vel_x_range = jnp.array(lin_vel_x_range)
        self.lin_vel_y_range = jnp.array(lin_vel_y_range)
        self.ang_vel_yaw_range = jnp.array(ang_vel_yaw_range)
        self.gait_frequency_range = jnp.array(gait_frequency_range)
        self.zero_command_probability = jnp.array(zero_command_probability)

        self.tracking_sigma = jnp.array(tracking_sigma)
        self.max_foot_height = jnp.array(max_foot_height)
        self.command_resample_steps = int(command_resample_steps)

        defaults = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.75,
            "ang_vel_xy": -0.15,
            "orientation": -2.0,
            "torques": 0.0,
            "action_rate": 0.0,
            "dof_acc": 0.0,
            "feet_slip": -0.25,
            "feet_air_time": 2.0,
            "feet_phase": 1.0,
            "alive": 0.0,
            "termination": -100.0,
            "stand_still": -1.0,
            "collision": -0.1,
            "dof_pos_limits": -1.0,
            "pose": -0.1,
            "joint_deviation_hip": -0.25,
            "joint_deviation_knee": -0.1,
        }
        weights = {**defaults, **(reward_weights or {})}

        self.reward_tracking_lin_vel = jnp.array(weights["tracking_lin_vel"])
        self.reward_tracking_ang_vel = jnp.array(weights["tracking_ang_vel"])
        self.reward_ang_vel_xy = jnp.array(weights["ang_vel_xy"])
        self.reward_orientation = jnp.array(weights["orientation"])
        self.reward_torques = jnp.array(weights["torques"])
        self.reward_action_rate = jnp.array(weights["action_rate"])
        self.reward_dof_acc = jnp.array(weights["dof_acc"])
        self.reward_feet_slip = jnp.array(weights["feet_slip"])
        self.reward_feet_air_time = jnp.array(weights["feet_air_time"])
        self.reward_feet_phase = jnp.array(weights["feet_phase"])
        self.reward_alive = jnp.array(weights["alive"])
        self.reward_termination = jnp.array(weights["termination"])
        self.reward_stand_still = jnp.array(weights["stand_still"])
        self.reward_collision = jnp.array(weights["collision"])
        self.reward_dof_pos_limits = jnp.array(weights["dof_pos_limits"])
        self.reward_pose = jnp.array(weights["pose"])
        self.reward_joint_deviation_hip = jnp.array(weights["joint_deviation_hip"])
        self.reward_joint_deviation_knee = jnp.array(weights["joint_deviation_knee"])

        # fmt: off
        self.pose_weights = jnp.array([
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # left leg
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # right leg
            1.0, 1.0, 1.0,                     # waist
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
        ])
        # fmt: on

        mj = self.mujoco_model
        hip_joint_names = ["hip_roll", "hip_yaw"]
        hip_indices = []
        for side in ["left", "right"]:
            for name in hip_joint_names:
                hip_indices.append(mj.joint(f"{side}_{name}_joint").qposadr - 7)
        self.hip_indices = tuple(hip_indices)

        knee_indices = []
        for side in ["left", "right"]:
            knee_indices.append(mj.joint(f"{side}_knee_joint").qposadr - 7)
        self.knee_indices = tuple(knee_indices)

    def initial(self, *, key: Key[Array, ""]) -> G1EnvState:
        randomize_key, qpos_key, qvel_key, cmd_key, freq_key = jr.split(key, 5)

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
            * jr.uniform(qpos_key, shape=(NUM_ACTUATED_DOFS,), minval=0.5, maxval=1.5)
        )
        qvel = self.init_qvel.at[:6].set(
            jr.uniform(qvel_key, shape=(6,), minval=-0.5, maxval=0.5)
        )

        data = mjx.make_data(model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=qpos[7:])
        data = mjx.forward(model, data)

        command = self.sample_command(key=cmd_key)
        gait_frequency = jr.uniform(
            freq_key,
            minval=self.gait_frequency_range[0],
            maxval=self.gait_frequency_range[1],
        )

        return G1EnvState(
            sim_state=data,
            t=jnp.array(0.0),
            model=model,
            last_action=jnp.zeros(NUM_ACTUATED_DOFS),
            gait_phase=gait.initial_gait_phase(),
            gait_frequency=gait_frequency,
            command=command,
            step_count=jnp.array(0.0),
            feet_air_time=jnp.zeros(2),
            last_contact=jnp.zeros(2, dtype=bool),
        )

    def observation(
        self, state: G1EnvState, *, key: Key[Array, ""]
    ) -> Float[Array, "103"]:
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

        phase_obs = gait.gait_phase_observation(state.gait_phase)

        return jnp.concatenate(
            [
                linvel,  # 3
                gyro,  # 3
                gravity,  # 3
                state.command,  # 3
                joint_angles,  # 29
                joint_vel,  # 29
                state.last_action,  # 29
                phase_obs,  # 4
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
        command = next_state.command

        local_vel = self._local_linvel(data)
        pelvis_gyro = self._gyro(data)
        torso_angvel = self._global_angvel(data, "torso")
        torso_gravity = self._torso_gravity_vector(data)

        lin_vel_error = jnp.sum(jnp.square(command[:2] - local_vel[:2]))
        tracking_lin = (
            jnp.exp(-lin_vel_error / self.tracking_sigma) * self.reward_tracking_lin_vel
        )

        ang_vel_error = jnp.square(command[2] - pelvis_gyro[2])
        tracking_ang = (
            jnp.exp(-ang_vel_error / self.tracking_sigma) * self.reward_tracking_ang_vel
        )

        ang_vel_xy = jnp.sum(jnp.square(torso_angvel[:2])) * self.reward_ang_vel_xy
        orientation = jnp.sum(jnp.square(torso_gravity[:2])) * self.reward_orientation

        torques = jnp.sum(jnp.abs(data.actuator_force)) * self.reward_torques
        action_rate = (
            jnp.sum(jnp.square(action - state.last_action)) * self.reward_action_rate
        )
        dof_acc = jnp.sum(jnp.square(data.qacc[6:])) * self.reward_dof_acc

        foot_contact = self._foot_contact(data)
        body_vel = self._global_linvel(data, "pelvis")[:2]
        feet_slip = (
            jnp.sum(jnp.linalg.norm(body_vel) * foot_contact) * self.reward_feet_slip
        )

        contact_filt = foot_contact | state.last_contact
        first_contact = (state.feet_air_time > 0.0) * contact_filt
        air_time_reward = jnp.clip(state.feet_air_time - 0.2, max=0.3) * first_contact
        feet_air_time = jnp.sum(air_time_reward) * self.reward_feet_air_time

        foot_pos = self._foot_positions(data)
        foot_z = foot_pos[:, 2]
        desired_z = gait.desired_foot_height(
            next_state.gait_phase, self.max_foot_height
        )
        phase_error = jnp.sum(jnp.square(foot_z - desired_z))
        body_angvel_z = self._global_angvel(data, "pelvis")[2]
        moving = jnp.logical_or(
            jnp.linalg.norm(body_vel) > 0.1, jnp.abs(body_angvel_z) > 0.1
        )
        moving = jnp.logical_or(moving, jnp.linalg.norm(command) > 0.01)
        feet_phase = jnp.exp(-phase_error / 0.01) * moving * self.reward_feet_phase

        alive = self.reward_alive

        done = self._is_fallen(data) | self._self_contact_termination(data)
        termination = done.astype(float) * self.reward_termination

        cmd_norm = jnp.linalg.norm(command)
        stand_still = (
            jnp.sum(jnp.abs(data.qpos[7:] - self.default_joint_positions))
            * (cmd_norm < 0.01)
            * self.reward_stand_still
        )

        collision = self._hand_collision(data).astype(float) * self.reward_collision

        out_of_limits = -jnp.clip(data.qpos[7:] - self.soft_joint_lower_limits, max=0.0)
        out_of_limits = out_of_limits + jnp.clip(
            data.qpos[7:] - self.soft_joint_upper_limits, min=0.0
        )
        dof_pos_limits = jnp.sum(out_of_limits) * self.reward_dof_pos_limits

        pose = (
            jnp.sum(jnp.square(data.qpos[7:] - self.default_joint_positions))
            * self.reward_pose
        )

        hip_error = (
            data.qpos[7:][jnp.array(self.hip_indices)]
            - self.default_joint_positions[jnp.array(self.hip_indices)]
        )
        hip_deviation = jnp.sum(jnp.abs(hip_error)) * self.reward_joint_deviation_hip

        knee_error = (
            data.qpos[7:][jnp.array(self.knee_indices)]
            - self.default_joint_positions[jnp.array(self.knee_indices)]
        )
        knee_deviation = jnp.sum(jnp.abs(knee_error)) * self.reward_joint_deviation_knee

        total = (
            tracking_lin
            + tracking_ang
            + ang_vel_xy
            + orientation
            + torques
            + action_rate
            + dof_acc
            + feet_slip
            + feet_air_time
            + feet_phase
            + alive
            + termination
            + stand_still
            + collision
            + dof_pos_limits
            + pose
            + hip_deviation
            + knee_deviation
        )
        return total * self.dt

    def terminal(self, state: G1EnvState, *, key: Key[Array, ""]) -> Bool[Array, ""]:
        data = state.sim_state
        fallen = self._is_fallen(data)
        nan_detected = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        self_contact = self._self_contact_termination(data)
        return fallen | nan_detected | self_contact

    def sample_command(self, *, key: Key[Array, ""]) -> Float[Array, "3"]:
        """Sample a random velocity command with chance of zero command."""
        vx_key, vy_key, yaw_key, zero_key = jr.split(key, 4)
        vx = jr.uniform(
            vx_key, minval=self.lin_vel_x_range[0], maxval=self.lin_vel_x_range[1]
        )
        vy = jr.uniform(
            vy_key, minval=self.lin_vel_y_range[0], maxval=self.lin_vel_y_range[1]
        )
        yaw = jr.uniform(
            yaw_key, minval=self.ang_vel_yaw_range[0], maxval=self.ang_vel_yaw_range[1]
        )
        cmd = jnp.array([vx, vy, yaw])
        return jnp.where(
            jr.bernoulli(zero_key, p=self.zero_command_probability),
            jnp.zeros(3),
            cmd,
        )

    def _is_fallen(self, data: mjx.Data) -> Bool[Array, ""]:
        """Robot has fallen if torso gravity z-axis points downward."""
        torso_gravity = self._torso_gravity_vector(data)
        return torso_gravity[2] < 0.0
