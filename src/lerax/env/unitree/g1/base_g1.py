"""Base class and state for Unitree G1 humanoid robot environments.

The G1 environments extend ``AbstractEnv`` directly rather than
``AbstractMujocoEnv`` because domain randomization requires the MJX
model to live on the *state* (so it can be traced/vmapped per-episode).
``AbstractMujocoEnv.transition`` uses ``self.model`` (a static field),
which cannot support per-episode randomized dynamics under JIT.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import equinox as eqx
import mujoco
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key
from mujoco import mjx

from lerax.render import AbstractRenderer
from lerax.render.mujoco_renderer import AbstractMujocoRenderer, MujocoRenderer
from lerax.space import Box

from ...base_env import AbstractEnv, AbstractEnvState
from ...mujoco.base_mujoco import tree_fix_dtype

NUM_ACTUATED_DOFS = 29
NUM_FREE_DOFS = 6


class G1EnvState(AbstractEnvState):
    """Environment state for all G1 tasks.

    Carries the per-episode randomized model alongside the physics state
    and locomotion tracking variables.

    Attributes:
        sim_state: MJX physics data.
        t: Simulation time in seconds.
        model: Per-episode randomized MJX model.
        last_action: Previous action for action rate penalty.
        gait_phase: Gait phase for [left, right] feet in [-pi, pi).
        gait_frequency: Gait frequency in Hz, sampled per episode.
        command: Velocity command [vx, vy, yaw_rate].
        step_count: Number of control steps taken.
        feet_air_time: Seconds each foot has been in the air.
        last_contact: Whether each foot was in contact last step.
    """

    sim_state: mjx.Data = eqx.field(converter=tree_fix_dtype)
    t: Float[Array, ""]
    model: mjx.Model
    last_action: Float[Array, "29"]
    gait_phase: Float[Array, "2"]
    gait_frequency: Float[Array, ""]
    command: Float[Array, "3"]
    step_count: Float[Array, ""]
    feet_air_time: Float[Array, "2"]
    last_contact: Bool[Array, "2"]


def _get_sensor_data(
    mj_model: mujoco.MjModel,
    data: mjx.Data,
    sensor_name: str,
) -> Float[Array, "..."]:
    """Read sensor data by name from MJX data."""
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


class AbstractG1Env(
    AbstractEnv[G1EnvState, Float[Array, "29"], Float[Array, "..."], None]
):
    """Abstract base for Unitree G1 humanoid environments.

    Provides the physics transition, sensor reading, and rendering shared
    across locomotion, standing, and standup tasks. Subclasses implement
    ``initial``, ``observation``, ``reward``, and ``terminal``.
    """

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[Box]
    observation_space: eqx.AbstractVar[Box]

    base_model: eqx.AbstractVar[mjx.Model]
    mujoco_model: eqx.AbstractVar[mujoco.MjModel]
    frame_skip: eqx.AbstractVar[int]
    dt: eqx.AbstractVar[Float[Array, ""]]

    action_scale: eqx.AbstractVar[Float[Array, ""]]
    default_joint_positions: eqx.AbstractVar[Float[Array, "29"]]

    init_qpos: eqx.AbstractVar[Float[Array, "..."]]
    init_qvel: eqx.AbstractVar[Float[Array, "..."]]

    nominal_friction_loss: eqx.AbstractVar[Float[Array, "..."]]
    nominal_armature: eqx.AbstractVar[Float[Array, "..."]]
    nominal_body_mass: eqx.AbstractVar[Float[Array, "..."]]
    torso_body_id: eqx.AbstractVar[int]

    friction_range: eqx.AbstractVar[tuple[float, float]]
    friction_loss_scale_range: eqx.AbstractVar[tuple[float, float]]
    armature_scale_range: eqx.AbstractVar[tuple[float, float]]
    mass_scale_range: eqx.AbstractVar[tuple[float, float]]
    torso_offset_range: eqx.AbstractVar[tuple[float, float]]

    push_enable: eqx.AbstractVar[bool]
    push_interval_range: eqx.AbstractVar[Float[Array, "2"]]
    push_magnitude_range: eqx.AbstractVar[Float[Array, "2"]]

    noise_level: eqx.AbstractVar[Float[Array, ""]]
    noise_scales: eqx.AbstractVar[dict[str, float]]

    soft_joint_pos_limit_factor: eqx.AbstractVar[Float[Array, ""]]
    joint_lower_limits: eqx.AbstractVar[Float[Array, "29"]]
    joint_upper_limits: eqx.AbstractVar[Float[Array, "29"]]
    soft_joint_lower_limits: eqx.AbstractVar[Float[Array, "29"]]
    soft_joint_upper_limits: eqx.AbstractVar[Float[Array, "29"]]

    pelvis_imu_site_id: eqx.AbstractVar[int]
    torso_imu_site_id: eqx.AbstractVar[int]

    feet_site_ids: eqx.AbstractVar[tuple[int, int]]
    foot_linvel_sensor_slices: eqx.AbstractVar[tuple[tuple[int, int], tuple[int, int]]]

    def action_mask(self, state: G1EnvState, *, key: Key[Array, ""]) -> None:
        return None

    def transition(
        self, state: G1EnvState, action: Float[Array, "29"], *, key: Key[Array, ""]
    ) -> G1EnvState:
        """Step physics with domain-randomized model.

        Computes motor targets from the action, optionally applies push
        perturbations, steps physics via ``lax.scan`` using the per-episode
        randomized model on the state, then updates gait phase and tracking.
        """
        from .gait import advance_gait_phase

        push_key, key = jr.split(key)

        motor_targets = self.default_joint_positions + action * self.action_scale

        data = state.sim_state.replace(ctrl=motor_targets)

        if self.push_enable:
            push_theta_key, push_mag_key = jr.split(push_key)
            push_theta = jr.uniform(push_theta_key, maxval=2 * jnp.pi)
            push_magnitude = jr.uniform(
                push_mag_key,
                minval=self.push_magnitude_range[0],
                maxval=self.push_magnitude_range[1],
            )
            push_direction = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
            push_interval_steps = jnp.round(
                (self.push_interval_range[0] + self.push_interval_range[1])
                / 2
                / self.dt
            ).astype(int)
            should_push = (
                jnp.mod(state.step_count.astype(int) + 1, push_interval_steps) == 0
            )
            push = push_direction * push_magnitude * should_push
            qvel = data.qvel.at[:2].add(push)
            data = data.replace(qvel=qvel)

        model = state.model

        def step_once(sim_data: mjx.Data, _: None) -> tuple[mjx.Data, None]:
            sim_data = mjx.step(model, sim_data)
            return sim_data, None

        data, _ = lax.scan(step_once, data, None, length=self.frame_skip)

        foot_contact = self._foot_contact(data)
        feet_air_time = (state.feet_air_time + self.dt) * ~foot_contact

        new_phase = advance_gait_phase(state.gait_phase, state.gait_frequency, self.dt)

        return G1EnvState(
            sim_state=data,
            t=state.t + self.dt,
            model=model,
            last_action=action,
            gait_phase=new_phase,
            gait_frequency=state.gait_frequency,
            command=state.command,
            step_count=state.step_count + 1,
            feet_air_time=feet_air_time,
            last_contact=foot_contact,
        )

    def truncate(self, state: G1EnvState) -> Bool[Array, ""]:
        return jnp.array(False)

    def transition_info(
        self,
        state: G1EnvState,
        action: Float[Array, "29"],
        next_state: G1EnvState,
    ) -> dict:
        return {}

    def state_info(self, state: G1EnvState) -> dict:
        return {}

    def default_renderer(self) -> MujocoRenderer:
        return MujocoRenderer(self.mujoco_model)

    def render(self, state: G1EnvState, renderer: AbstractRenderer):
        if not isinstance(renderer, AbstractMujocoRenderer):
            raise TypeError("G1 environment requires a Mujoco renderer.")
        renderer.render(state.sim_state)
        renderer.draw()

    def _local_linvel(self, data: mjx.Data) -> Float[Array, "3"]:
        """Local linear velocity from pelvis IMU velocimeter."""
        return _get_sensor_data(self.mujoco_model, data, "local_linvel_pelvis")

    def _gyro(self, data: mjx.Data) -> Float[Array, "3"]:
        """Angular velocity from pelvis gyroscope."""
        return _get_sensor_data(self.mujoco_model, data, "gyro_pelvis")

    def _gravity_vector(self, data: mjx.Data) -> Float[Array, "3"]:
        """Gravity direction in pelvis frame (upvector sensor)."""
        return data.site_xmat[self.pelvis_imu_site_id].reshape(3, 3).T @ jnp.array(
            [0.0, 0.0, -1.0]
        )

    def _torso_gravity_vector(self, data: mjx.Data) -> Float[Array, "3"]:
        """Gravity direction in torso frame."""
        return _get_sensor_data(self.mujoco_model, data, "upvector_torso")

    def _global_linvel(self, data: mjx.Data, site: str = "pelvis") -> Float[Array, "3"]:
        """Global linear velocity from frame sensor."""
        return _get_sensor_data(self.mujoco_model, data, f"global_linvel_{site}")

    def _global_angvel(self, data: mjx.Data, site: str = "pelvis") -> Float[Array, "3"]:
        """Global angular velocity from frame sensor."""
        return _get_sensor_data(self.mujoco_model, data, f"global_angvel_{site}")

    def _joint_angles_offset(self, data: mjx.Data) -> Float[Array, "29"]:
        """Joint angles relative to default pose."""
        return data.qpos[7:] - self.default_joint_positions

    def _joint_velocities(self, data: mjx.Data) -> Float[Array, "29"]:
        """Actuated joint velocities."""
        return data.qvel[6:]

    def _foot_contact(self, data: mjx.Data) -> Bool[Array, "2"]:
        """Binary foot contact flags from contact sensors."""
        mj = self.mujoco_model
        left_sensor_id = mj.sensor("left_foot_floor_found").id
        right_sensor_id = mj.sensor("right_foot_floor_found").id
        left_contact = data.sensordata[mj.sensor_adr[left_sensor_id]] > 0
        right_contact = data.sensordata[mj.sensor_adr[right_sensor_id]] > 0
        return jnp.array([left_contact, right_contact])

    def _self_contact_termination(self, data: mjx.Data) -> Bool[Array, ""]:
        """Check for dangerous self-contacts (foot-foot, foot-shin)."""
        mj = self.mujoco_model
        foot_foot = (
            data.sensordata[mj.sensor_adr[mj.sensor("right_foot_left_foot_found").id]]
            > 0
        )
        left_right_shin = (
            data.sensordata[mj.sensor_adr[mj.sensor("left_foot_right_shin_found").id]]
            > 0
        )
        right_left_shin = (
            data.sensordata[mj.sensor_adr[mj.sensor("right_foot_left_shin_found").id]]
            > 0
        )
        return foot_foot | left_right_shin | right_left_shin

    def _hand_collision(self, data: mjx.Data) -> Bool[Array, ""]:
        """Check for hand-thigh collisions."""
        mj = self.mujoco_model
        left = (
            data.sensordata[mj.sensor_adr[mj.sensor("left_hand_left_thigh_found").id]]
            > 0
        )
        right = (
            data.sensordata[mj.sensor_adr[mj.sensor("right_hand_right_thigh_found").id]]
            > 0
        )
        return left | right

    def _foot_positions(self, data: mjx.Data) -> Float[Array, "2 3"]:
        """World-frame positions of foot sites."""
        left_id, right_id = self.feet_site_ids
        return jnp.stack([data.site_xpos[left_id], data.site_xpos[right_id]])

    def _foot_velocities(self, data: mjx.Data) -> Float[Array, "2 3"]:
        """Global linear velocities of foot sites."""
        (l_start, l_end), (r_start, r_end) = self.foot_linvel_sensor_slices
        left_vel = data.sensordata[l_start:l_end]
        right_vel = data.sensordata[r_start:r_end]
        return jnp.stack([left_vel, right_vel])

    @abstractmethod
    def sample_command(self, *, key: Key[Array, ""]) -> Float[Array, "3"]:
        """Sample a velocity command for a new episode."""

    def _init_common(
        self,
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
    ):
        """Shared initialization for all G1 environments."""
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(f"G1 asset not found: {asset_path}")

        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))

        self.base_model = mjx.put_model(mj_model)
        self.mujoco_model = mj_model

        sim_dt = mj_model.opt.timestep
        self.frame_skip = round(1.0 / (control_frequency_hz * sim_dt))
        self.dt = jnp.array(sim_dt * self.frame_skip)

        keyframe = mj_model.keyframe(keyframe_name)
        self.default_joint_positions = jnp.array(keyframe.qpos[7:])
        self.init_qpos = jnp.array(keyframe.qpos)
        self.init_qvel = jnp.zeros(mj_model.nv)

        self.action_scale = jnp.array(action_scale)

        self.nominal_friction_loss = jnp.array(mj_model.dof_frictionloss[6:])
        self.nominal_armature = jnp.array(mj_model.dof_armature[6:])
        self.nominal_body_mass = jnp.array(mj_model.body_mass)
        self.torso_body_id = mj_model.body("torso_link").id

        self.friction_range = friction_range
        self.friction_loss_scale_range = friction_loss_scale_range
        self.armature_scale_range = armature_scale_range
        self.mass_scale_range = mass_scale_range
        self.torso_offset_range = torso_offset_range

        self.push_enable = push_enable
        self.push_interval_range = jnp.array(push_interval_range)
        self.push_magnitude_range = jnp.array(push_magnitude_range)

        self.noise_level = jnp.array(noise_level)
        self.noise_scales = (
            noise_scales
            if noise_scales is not None
            else {
                "joint_pos": 0.03,
                "joint_vel": 1.5,
                "gravity": 0.05,
                "linvel": 0.1,
                "gyro": 0.2,
            }
        )

        lowers, uppers = mj_model.jnt_range[1:].T
        self.joint_lower_limits = jnp.array(lowers)
        self.joint_upper_limits = jnp.array(uppers)
        self.soft_joint_pos_limit_factor = jnp.array(soft_joint_pos_limit_factor)
        center = (lowers + uppers) / 2
        half_range = (uppers - lowers) / 2
        self.soft_joint_lower_limits = jnp.array(
            center - half_range * soft_joint_pos_limit_factor
        )
        self.soft_joint_upper_limits = jnp.array(
            center + half_range * soft_joint_pos_limit_factor
        )

        self.pelvis_imu_site_id = mj_model.site("imu_in_pelvis").id
        self.torso_imu_site_id = mj_model.site("imu_in_torso").id

        left_foot_site = mj_model.site("left_foot").id
        right_foot_site = mj_model.site("right_foot").id
        self.feet_site_ids = (left_foot_site, right_foot_site)

        def _sensor_slice(name: str) -> tuple[int, int]:
            sid = mj_model.sensor(name).id
            adr = mj_model.sensor_adr[sid]
            dim = mj_model.sensor_dim[sid]
            return (int(adr), int(adr + dim))

        self.foot_linvel_sensor_slices = (
            _sensor_slice("left_foot_global_linvel"),
            _sensor_slice("right_foot_global_linvel"),
        )

        self.action_space = Box(low=-1.0, high=1.0, shape=(NUM_ACTUATED_DOFS,))
