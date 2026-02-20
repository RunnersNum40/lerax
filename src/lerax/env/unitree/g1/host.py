"""G1 HoST environment: stand up from fallen postures.

Implements the reward structure from "Learning Humanoid Standing-up Control
across Diverse Postures" (HoST, RSS 2025). The agent starts from a supine
or prone pose and must stand upright using a multi-group reward combining
task progress, regularization, style, and post-standing balance terms.

Reference: https://arxiv.org/abs/2502.08378
"""

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

OBSERVATION_SIZE = 94


def _tolerance(
    value: Float[Array, ""],
    lower: float,
    margin: float,
    value_at_margin: float,
) -> Float[Array, ""]:
    """Gaussian-bell tolerance reward (lower-bounded, no upper bound).

    Returns 1.0 when ``value >= lower``, decays smoothly below using a
    Gaussian with the specified margin and value at the margin boundary.
    """
    distance = jnp.maximum(lower - value, 0.0)
    scale = jnp.sqrt(-2 * jnp.log(value_at_margin)) / margin
    return jnp.where(distance < 1e-6, 1.0, jnp.exp(-0.5 * (distance * scale) ** 2))


class G1HoST(AbstractG1Env):
    """Unitree G1 HoST: stand up from fallen postures.

    The agent starts from a random supine (on back) or prone (face down)
    pose and must stand upright. Uses the HoST reward structure with four
    groups: task (multiplicative orientation + height), regularization
    (motion smoothness), style (joint deviation penalties), and target
    (post-standing balance).

    Observation (94 dims):
        - Angular velocity × 0.25 (3)
        - Projected gravity in pelvis frame (3)
        - Raw joint positions (29)
        - Joint velocities × 0.05 (29)
        - Last action (29)
        - Action rescale beta (1)

    Action: [-1, 1]^29 applied relative to current joint position:
        ``target = current_pos + action * beta``
    """

    name: ClassVar[str] = "G1HoST"

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

    relative_actions: bool
    pulling_force_magnitude: Float[Array, ""]
    unactuated_steps: int

    task_weight: Float[Array, ""]
    regularization_weight: Float[Array, ""]
    style_weight: Float[Array, ""]
    target_weight: Float[Array, ""]

    left_knee_body_id: int
    right_knee_body_id: int

    waist_yaw_index: int
    hip_yaw_indices: tuple[int, int]
    hip_roll_indices: tuple[int, int]
    left_shoulder_roll_index: int
    right_shoulder_roll_index: int
    knee_indices: tuple[int, int]
    upper_body_indices: tuple[int, ...]
    upper_body_targets: Float[Array, "..."]

    def __init__(
        self,
        *,
        xml_file: str | Path = "scene_mjx.xml",
        control_frequency_hz: float = 50.0,
        action_scale: float = 0.3,
        keyframe_name: str = "knees_bent",
        soft_joint_pos_limit_factor: float = 0.95,
        push_enable: bool = False,
        push_interval_range: tuple[float, float] = (5.0, 10.0),
        push_magnitude_range: tuple[float, float] = (0.1, 2.0),
        noise_level: float = 1.0,
        noise_scales: dict[str, float] | None = None,
        friction_range: tuple[float, float] = (0.1, 1.0),
        friction_loss_scale_range: tuple[float, float] = (0.5, 2.0),
        armature_scale_range: tuple[float, float] = (1.0, 1.05),
        mass_scale_range: tuple[float, float] = (0.8, 1.2),
        torso_offset_range: tuple[float, float] = (-2.0, 5.0),
        pulling_force_magnitude: float = 0.0,
        unactuated_steps: int = 30,
        task_weight: float = 2.5,
        regularization_weight: float = 0.1,
        style_weight: float = 1.0,
        target_weight: float = 1.0,
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
            relative_actions=True,
            pulling_force_magnitude=pulling_force_magnitude,
            unactuated_steps=unactuated_steps,
        )

        self.observation_space = Box(
            low=-jnp.inf, high=jnp.inf, shape=(OBSERVATION_SIZE,)
        )

        self.task_weight = jnp.array(task_weight)
        self.regularization_weight = jnp.array(regularization_weight)
        self.style_weight = jnp.array(style_weight)
        self.target_weight = jnp.array(target_weight)

        mj = self.mujoco_model

        self.left_knee_body_id = int(mj.body("left_knee_link").id)
        self.right_knee_body_id = int(mj.body("right_knee_link").id)

        def _dof_index(joint_name: str) -> int:
            return int(mj.joint(joint_name).qposadr.item()) - 7

        self.waist_yaw_index = _dof_index("waist_yaw_joint")
        self.hip_yaw_indices = (
            _dof_index("left_hip_yaw_joint"),
            _dof_index("right_hip_yaw_joint"),
        )
        self.hip_roll_indices = (
            _dof_index("left_hip_roll_joint"),
            _dof_index("right_hip_roll_joint"),
        )
        self.left_shoulder_roll_index = _dof_index("left_shoulder_roll_joint")
        self.right_shoulder_roll_index = _dof_index("right_shoulder_roll_joint")
        self.knee_indices = (
            _dof_index("left_knee_joint"),
            _dof_index("right_knee_joint"),
        )

        upper_start = _dof_index("waist_yaw_joint")
        self.upper_body_indices = tuple(range(upper_start, NUM_ACTUATED_DOFS))
        self.upper_body_targets = self.default_joint_positions[
            jnp.array(self.upper_body_indices)
        ]

    def initial(self, *, key: Key[Array, ""]) -> G1EnvState:
        """Initialize from a random supine or prone fallen posture."""
        randomize_key, pose_key, scale_key, offset_key = jr.split(key, 4)

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

        # Supine (on back) vs prone (face down)
        supine = jr.bernoulli(pose_key)
        supine_quat = jnp.array([1.0, 0.0, -1.0, 0.0])
        supine_quat = supine_quat / jnp.linalg.norm(supine_quat)
        prone_quat = jnp.array([1.0, 0.0, 1.0, 0.0])
        prone_quat = prone_quat / jnp.linalg.norm(prone_quat)
        base_quat = jnp.where(supine, supine_quat, prone_quat)

        # Randomize joints: default * U(0.9, 1.1) + U(-0.1, 0.1), clipped
        noise_scale = jr.uniform(
            scale_key, shape=(NUM_ACTUATED_DOFS,), minval=0.9, maxval=1.1
        )
        noise_offset = jr.uniform(
            offset_key, shape=(NUM_ACTUATED_DOFS,), minval=-0.1, maxval=0.1
        )
        joint_pos = self.default_joint_positions * noise_scale + noise_offset
        joint_pos = jnp.clip(
            joint_pos, self.soft_joint_lower_limits, self.soft_joint_upper_limits
        )

        qpos = jnp.zeros_like(self.init_qpos)
        qpos = qpos.at[3:7].set(base_quat)
        qpos = qpos.at[7:].set(joint_pos)
        qvel = jnp.zeros_like(self.init_qvel)

        data = mjx.make_data(model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=joint_pos)
        data = mjx.forward(model, data)
        data = self._snap_to_ground(model, data)

        return G1EnvState(
            sim_state=data,
            t=jnp.array(0.0),
            model=model,
            last_action=jnp.zeros(NUM_ACTUATED_DOFS),
            last_last_action=jnp.zeros(NUM_ACTUATED_DOFS),
            gait_phase=initial_gait_phase(),
            gait_frequency=jnp.array(0.0),
            command=jnp.zeros(3),
            step_count=jnp.array(0.0),
            feet_air_time=jnp.zeros(2),
            last_contact=jnp.zeros(2, dtype=bool),
        )

    def observation(
        self, state: G1EnvState, *, key: Key[Array, ""]
    ) -> Float[Array, "94"]:
        data = state.sim_state

        angular_vel = self._gyro(data) * 0.25
        projected_gravity = self._gravity_vector(data)
        joint_pos = data.qpos[7:]
        joint_vel = self._joint_velocities(data) * 0.05
        last_action = state.last_action
        beta = jnp.array([self.action_scale])

        obs = jnp.concatenate(
            [angular_vel, projected_gravity, joint_pos, joint_vel, last_action, beta]
        )

        if self.unactuated_steps > 0:
            settling = state.step_count < self.unactuated_steps
            obs = jnp.where(settling, jnp.zeros_like(obs), obs)

        return obs

    def reward(
        self,
        state: G1EnvState,
        action: Float[Array, "29"],
        next_state: G1EnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        data = next_state.sim_state
        prev_data = state.sim_state

        gravity = self._gravity_vector(data)
        base_z = data.qpos[2]
        dof_pos = data.qpos[7:]
        dof_vel = data.qvel[6:]
        prev_dof_vel = prev_data.qvel[6:]
        torques = data.actuator_force
        foot_pos = self._foot_positions(data)
        base_xy = data.qpos[:2]

        torso_height = base_z - jnp.min(foot_pos[:, 2])

        # --- Task group (multiplicative) ---
        task_orientation = _tolerance(
            -gravity[2], lower=0.99, margin=1.0, value_at_margin=0.05
        )
        task_height = _tolerance(
            torso_height, lower=1.0, margin=1.0, value_at_margin=0.1
        )
        task_reward = task_orientation * task_height * self.task_weight

        # --- Regularization group (additive × dt) ---
        dof_acc = (prev_dof_vel - dof_vel) / self.dt
        reg_dof_acc = -2.5e-7 * jnp.sum(dof_acc**2)
        reg_action_rate = -0.01 * jnp.sum((state.last_action - action) ** 2)
        reg_smoothness = -0.01 * jnp.sum(
            (action - 2 * state.last_action + state.last_last_action) ** 2
        )
        reg_torques = -2.5e-6 * jnp.sum(torques**2)
        reg_joint_power = -2.5e-5 * jnp.sum(jnp.abs(dof_vel) * jnp.abs(torques))
        reg_dof_vel = -1e-3 * jnp.sum(dof_vel**2)
        reg_tracking = -0.00025 * jnp.sum((data.ctrl - dof_pos) ** 2)

        limit_violation = jnp.maximum(
            dof_pos - self.soft_joint_upper_limits, 0.0
        ) + jnp.maximum(self.soft_joint_lower_limits - dof_pos, 0.0)
        reg_pos_limits = -100.0 * jnp.sum(limit_violation)

        reg_total = (
            (
                reg_dof_acc
                + reg_action_rate
                + reg_smoothness
                + reg_torques
                + reg_joint_power
                + reg_dof_vel
                + reg_tracking
                + reg_pos_limits
            )
            * self.dt
            * self.regularization_weight
        )

        # --- Style group (additive × dt) ---
        waist_yaw = dof_pos[self.waist_yaw_index]
        style_waist = -10.0 * (jnp.abs(waist_yaw) > 1.4).astype(float)

        hip_yaw = dof_pos[jnp.array(self.hip_yaw_indices)]
        style_hip_yaw = -10.0 * (
            (jnp.max(jnp.abs(hip_yaw)) > 1.4) | (jnp.min(jnp.abs(hip_yaw)) > 0.9)
        ).astype(float)

        hip_roll = dof_pos[jnp.array(self.hip_roll_indices)]
        style_hip_roll = -10.0 * (
            (jnp.max(jnp.abs(hip_roll)) > 1.4) | (jnp.min(jnp.abs(hip_roll)) > 0.9)
        ).astype(float)

        left_shoulder = dof_pos[self.left_shoulder_roll_index]
        right_shoulder = dof_pos[self.right_shoulder_roll_index]
        style_shoulder = -2.5 * (
            (left_shoulder < -0.02) | (right_shoulder > 0.02)
        ).astype(float)

        left_foot_xy_dist = jnp.linalg.norm(foot_pos[0, :2] - base_xy)
        right_foot_xy_dist = jnp.linalg.norm(foot_pos[1, :2] - base_xy)
        standing_gate = (base_z > 0.65).astype(float)
        style_foot_left = (
            2.5
            * jnp.exp(-2.0 * jnp.maximum(left_foot_xy_dist, 0.3))
            * (foot_pos[0, 2] < 0.3).astype(float)
            * standing_gate
        )
        style_foot_right = (
            2.5
            * jnp.exp(-2.0 * jnp.maximum(right_foot_xy_dist, 0.3))
            * (foot_pos[1, 2] < 0.3).astype(float)
            * standing_gate
        )

        knee_vals = dof_pos[jnp.array(self.knee_indices)]
        style_knee = -0.25 * (
            (jnp.max(jnp.abs(knee_vals)) > 2.85) | (jnp.min(knee_vals) < -0.06)
        ).astype(float)

        left_shank_z = data.xmat[self.left_knee_body_id].reshape(3, 3)[2, 2]
        right_shank_z = data.xmat[self.right_knee_body_id].reshape(3, 3)[2, 2]
        shank_vert = (left_shank_z + right_shank_z) / 2.0
        style_shank = (
            10.0
            * _tolerance(shank_vert, lower=0.8, margin=1.0, value_at_margin=0.1)
            * (base_z > 0.45).astype(float)
        )

        feet_z_var = jnp.abs(foot_pos[0, 2] - foot_pos[1, 2])
        feet_z_max = jnp.maximum(foot_pos[0, 2], foot_pos[1, 2])
        style_ground_parallel = 20.0 * (
            (feet_z_var < 0.05) & (feet_z_max < 0.1)
        ).astype(float)

        feet_dist = jnp.linalg.norm(foot_pos[0] - foot_pos[1])
        style_feet_dist = -10.0 * (feet_dist > 0.9).astype(float)

        angvel = self._gyro(data)
        style_angvel_xy = (
            1.0
            * jnp.exp(-2.0 * jnp.sum(angvel[:2] ** 2))
            * (base_z > 0.45).astype(float)
        )

        style_total = (
            (
                style_waist
                + style_hip_yaw
                + style_hip_roll
                + style_shoulder
                + style_foot_left
                + style_foot_right
                + style_knee
                + style_shank
                + style_ground_parallel
                + style_feet_dist
                + style_angvel_xy
            )
            * self.dt
            * self.style_weight
        )

        # --- Target group (additive × dt, gated by base_z > 0.65) ---
        linvel = self._local_linvel(data)
        target_angvel_xy = 10.0 * jnp.exp(-2.0 * jnp.sum(angvel[:2] ** 2))
        target_linvel_xy = 10.0 * jnp.exp(-5.0 * jnp.sum(linvel[:2] ** 2))

        feet_height_diff = jnp.abs(foot_pos[0, 2] - foot_pos[1, 2]) * 10.0
        target_feet_var = 2.5 * jnp.exp(-2.0 * jnp.maximum(feet_height_diff, 0.2))

        upper_dof = dof_pos[jnp.array(self.upper_body_indices)]
        target_upper_dof = 10.0 * jnp.exp(
            -0.1 * jnp.sum((upper_dof - self.upper_body_targets) ** 2)
        )

        target_orientation = 10.0 * jnp.exp(-5.0 * jnp.sum(gravity[:2] ** 2))
        target_base_height = 10.0 * jnp.exp(-20.0 * jnp.abs(base_z - 0.75))

        target_total = (
            (
                target_angvel_xy
                + target_linvel_xy
                + target_feet_var
                + target_upper_dof
                + target_orientation
                + target_base_height
            )
            * self.dt
            * self.target_weight
            * standing_gate
        )

        return task_reward + reg_total + style_total + target_total

    def terminal(self, state: G1EnvState, *, key: Key[Array, ""]) -> Bool[Array, ""]:
        """Terminate on extreme velocity or NaN (after settling period)."""
        data = state.sim_state
        past_settling = state.step_count >= self.unactuated_steps

        dof_vel_exceeded = jnp.any(jnp.abs(data.qvel[6:]) > 300.0)
        base_vel_exceeded = jnp.linalg.norm(data.qvel[:3]) > 20.0
        nan_detected = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()

        velocity_termination = (dof_vel_exceeded | base_vel_exceeded) & past_settling
        return velocity_termination | nan_detected

    def sample_command(self, *, key: Key[Array, ""]) -> Float[Array, "3"]:
        """HoST uses zero velocity command."""
        return jnp.zeros(3)
