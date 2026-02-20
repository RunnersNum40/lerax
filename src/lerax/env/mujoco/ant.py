from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import mujoco
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key
from mujoco import mjx

from lerax.space import Box

from .base_mujoco import AbstractMujocoEnv, MujocoEnvState


class Ant(AbstractMujocoEnv[Float[Array, "..."], Float[Array, "..."]]):
    """MJX-based ant environment matching Gymnasium's Ant-v5."""

    name: ClassVar[str] = "Ant"

    action_space: Box
    observation_space: Box
    model: mjx.Model
    mujoco_model: mujoco.MjModel

    frame_skip: int
    dt: Float[Array, ""]

    forward_reward_weight: Float[Array, ""]
    ctrl_cost_weight: Float[Array, ""]
    contact_cost_weight: Float[Array, ""]
    contact_force_range: Float[Array, "2"]
    healthy_reward: Float[Array, ""]
    terminate_when_unhealthy: bool
    healthy_z_range: Float[Array, "2"]

    reset_noise_scale: Float[Array, ""]
    exclude_current_positions_from_observation: bool
    include_cfrc_ext_in_observation: bool

    main_body_id: int

    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    def __init__(
        self,
        *,
        xml_file: str | Path = "ant.xml",
        frame_skip: int = 5,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
    ):
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(f"Ant asset not found: {asset_path}")

        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        mj_data = mujoco.MjData(mj_model)

        self.model = mjx.put_model(mj_model)
        self.mujoco_model = mj_model

        self.frame_skip = int(frame_skip)
        self.dt = jnp.array(mj_model.opt.timestep * self.frame_skip)

        self.init_qpos = jnp.asarray(mj_data.qpos).reshape(-1)
        self.init_qvel = jnp.asarray(mj_data.qvel).reshape(-1)

        self.forward_reward_weight = jnp.array(forward_reward_weight)
        self.ctrl_cost_weight = jnp.array(ctrl_cost_weight)
        self.contact_cost_weight = jnp.array(contact_cost_weight)
        self.contact_force_range = jnp.asarray(contact_force_range)
        self.healthy_reward = jnp.array(healthy_reward)
        self.terminate_when_unhealthy = bool(terminate_when_unhealthy)
        self.healthy_z_range = jnp.asarray(healthy_z_range)

        self.reset_noise_scale = jnp.array(reset_noise_scale)
        self.exclude_current_positions_from_observation = bool(
            exclude_current_positions_from_observation
        )
        self.include_cfrc_ext_in_observation = bool(include_cfrc_ext_in_observation)

        self.main_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        obs_size = mj_data.qpos.size + mj_data.qvel.size
        obs_size -= 2 if self.exclude_current_positions_from_observation else 0
        obs_size += (
            mj_data.cfrc_ext[1:].size if self.include_cfrc_ext_in_observation else 0
        )

        bounds = jnp.array(mj_model.actuator_ctrlrange, dtype=jnp.float32)
        low, high = bounds[:, 0], bounds[:, 1]
        self.action_space = Box(low=low, high=high)

        high_obs = jnp.full((obs_size,), jnp.inf, dtype=jnp.float32)
        self.observation_space = Box(low=-high_obs, high=high_obs)

    def initial(self, *, key: Key[Array, ""]) -> MujocoEnvState:
        qpos_key, qvel_key = jr.split(key)

        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + jr.uniform(
            qpos_key, shape=self.init_qpos.shape, minval=noise_low, maxval=noise_high
        )
        qvel = self.init_qvel + self.reset_noise_scale * jr.normal(
            qvel_key, shape=self.init_qvel.shape
        )

        data = mjx.make_data(self.model)
        data = data.replace(qpos=qpos, qvel=qvel)

        return MujocoEnvState(sim_state=data, t=jnp.array(0.0))

    def observation(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Float[Array, "..."]:
        data = state.sim_state

        position = data.qpos.reshape(-1)
        velocity = data.qvel.reshape(-1)

        if self.exclude_current_positions_from_observation:
            position = position[2:]

        if self.include_cfrc_ext_in_observation:
            contact_forces = self.clipped_contact_forces(data)[1:].reshape(-1)
        else:
            contact_forces = jnp.zeros((0,), dtype=position.dtype)

        return jnp.concatenate((position, velocity, contact_forces))

    def reward(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        xy_before = state.sim_state.xpos[self.main_body_id][:2]
        xy_after = next_state.sim_state.xpos[self.main_body_id][:2]
        xy_velocity = (xy_after - xy_before) / self.dt
        x_velocity = xy_velocity[0]

        forward_reward = self.forward_reward_weight * x_velocity
        healthy_reward = (
            self.is_healthy(next_state.sim_state).astype(float) * self.healthy_reward
        )
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = self.contact_cost(next_state.sim_state)

        return forward_reward + healthy_reward - ctrl_cost - contact_cost

    def terminal(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        if not self.terminate_when_unhealthy:
            return jnp.array(False)
        return ~self.is_healthy(state.sim_state)

    def state_info(self, state: MujocoEnvState) -> dict:
        data = state.sim_state
        return {
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "distance_from_origin": jnp.linalg.norm(data.qpos[0:2], ord=2),
        }

    def transition_info(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
    ) -> dict:
        data = next_state.sim_state

        xy_before = state.sim_state.xpos[self.main_body_id][:2]
        xy_after = data.xpos[self.main_body_id][:2]
        xy_velocity = (xy_after - xy_before) / self.dt
        x_velocity, y_velocity = xy_velocity[0], xy_velocity[1]

        forward_reward = self.forward_reward_weight * x_velocity
        healthy_reward = self.is_healthy(data).astype(float) * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = self.contact_cost(data)

        return {
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "distance_from_origin": jnp.linalg.norm(data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

    def is_healthy(self, data: mjx.Data) -> Bool[Array, ""]:
        min_z, max_z = self.healthy_z_range
        z = data.qpos[2]
        state = jnp.concatenate([data.qpos, data.qvel])
        is_finite = jnp.all(jnp.isfinite(state))
        is_in_range = (z >= min_z) & (z <= max_z)
        return is_finite & is_in_range

    def clipped_contact_forces(self, data: mjx.Data) -> Float[Array, "..."]:
        min_val, max_val = self.contact_force_range
        return jnp.clip(data.cfrc_ext, min_val, max_val)

    def contact_cost(self, data: mjx.Data) -> Float[Array, ""]:
        contact_forces = self.clipped_contact_forces(data)
        return self.contact_cost_weight * jnp.sum(jnp.square(contact_forces))
