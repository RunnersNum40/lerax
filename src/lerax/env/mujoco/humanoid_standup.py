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


class HumanoidStandup(AbstractMujocoEnv[Float[Array, "..."], Float[Array, "..."]]):
    """MJX-based humanoid standup environment matching Gymnasium's HumanoidStandup-v5."""

    name: ClassVar[str] = "HumanoidStandup"

    action_space: Box
    observation_space: Box
    model: mjx.Model
    mujoco_model: mujoco.MjModel

    frame_skip: int
    dt: Float[Array, ""]

    uph_cost_weight: Float[Array, ""]
    ctrl_cost_weight: Float[Array, ""]
    impact_cost_weight: Float[Array, ""]
    impact_cost_range: Float[Array, "2"]

    reset_noise_scale: Float[Array, ""]
    exclude_current_positions_from_observation: bool
    include_cinert_in_observation: bool
    include_cvel_in_observation: bool
    include_qfrc_actuator_in_observation: bool
    include_cfrc_ext_in_observation: bool

    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    qpos_size: int
    qvel_size: int
    cinert_size: int
    cvel_size: int
    qfrc_actuator_size: int
    cfrc_ext_size: int

    def __init__(
        self,
        *,
        xml_file: str | Path = "humanoidstandup.xml",
        frame_skip: int = 5,
        uph_cost_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        impact_cost_weight: float = 0.5e-6,
        impact_cost_range: tuple[float, float] = (-jnp.inf, 10.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
    ):
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(f"HumanoidStandup asset not found: {asset_path}")

        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        mj_data = mujoco.MjData(mj_model)

        self.model = mjx.put_model(mj_model)
        self.mujoco_model = mj_model

        self.frame_skip = int(frame_skip)
        self.dt = jnp.array(mj_model.opt.timestep * self.frame_skip)

        self.init_qpos = jnp.asarray(mj_data.qpos).reshape(-1)
        self.init_qvel = jnp.asarray(mj_data.qvel).reshape(-1)

        self.uph_cost_weight = jnp.array(uph_cost_weight)
        self.ctrl_cost_weight = jnp.array(ctrl_cost_weight)
        self.impact_cost_weight = jnp.array(impact_cost_weight)
        self.impact_cost_range = jnp.asarray(impact_cost_range)

        self.reset_noise_scale = jnp.array(reset_noise_scale)
        self.exclude_current_positions_from_observation = bool(
            exclude_current_positions_from_observation
        )
        self.include_cinert_in_observation = bool(include_cinert_in_observation)
        self.include_cvel_in_observation = bool(include_cvel_in_observation)
        self.include_qfrc_actuator_in_observation = bool(
            include_qfrc_actuator_in_observation
        )
        self.include_cfrc_ext_in_observation = bool(include_cfrc_ext_in_observation)

        qpos_size = mj_data.qpos.size
        qvel_size = mj_data.qvel.size
        cinert_size = mj_data.cinert[1:].size
        cvel_size = mj_data.cvel[1:].size
        qfrc_actuator_size = mj_data.qvel.size - 6
        cfrc_ext_size = mj_data.cfrc_ext[1:].size

        skipped_qpos = 2 if self.exclude_current_positions_from_observation else 0

        obs_size = qpos_size + qvel_size - skipped_qpos
        obs_size += cinert_size if self.include_cinert_in_observation else 0
        obs_size += cvel_size if self.include_cvel_in_observation else 0
        obs_size += (
            qfrc_actuator_size if self.include_qfrc_actuator_in_observation else 0
        )
        obs_size += cfrc_ext_size if self.include_cfrc_ext_in_observation else 0

        self.qpos_size = qpos_size
        self.qvel_size = qvel_size
        self.cinert_size = cinert_size
        self.cvel_size = cvel_size
        self.qfrc_actuator_size = qfrc_actuator_size
        self.cfrc_ext_size = cfrc_ext_size

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
        qvel = self.init_qvel + jr.uniform(
            qvel_key, shape=self.init_qvel.shape, minval=noise_low, maxval=noise_high
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

        if self.include_cinert_in_observation:
            com_inertia = data.cinert[1:].reshape(-1)
        else:
            com_inertia = jnp.zeros((0,), dtype=position.dtype)

        if self.include_cvel_in_observation:
            com_velocity = data.cvel[1:].reshape(-1)
        else:
            com_velocity = jnp.zeros((0,), dtype=position.dtype)

        if self.include_qfrc_actuator_in_observation:
            actuator_forces = data.qfrc_actuator[6:].reshape(-1)
        else:
            actuator_forces = jnp.zeros((0,), dtype=position.dtype)

        if self.include_cfrc_ext_in_observation:
            external_contact_forces = data.cfrc_ext[1:].reshape(-1)
        else:
            external_contact_forces = jnp.zeros((0,), dtype=position.dtype)

        return jnp.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def reward(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        data = next_state.sim_state

        uph_cost = self.uph_cost_weight * (data.qpos[2] / self.dt)
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(data.ctrl))
        impact_cost = self.impact_cost(data)

        return uph_cost - ctrl_cost - impact_cost + 1

    def terminal(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return jnp.array(False)

    def state_info(self, state: MujocoEnvState) -> dict:
        data = state.sim_state
        return {
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "tendon_length": data.ten_length,
            "tendon_velocity": data.ten_velocity,
            "distance_from_origin": jnp.linalg.norm(data.qpos[0:2], ord=2),
        }

    def transition_info(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
    ) -> dict:
        data = next_state.sim_state

        uph_cost = self.uph_cost_weight * (data.qpos[2] / self.dt)
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(data.ctrl))
        impact_cost = self.impact_cost(data)

        return {
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "tendon_length": data.ten_length,
            "tendon_velocity": data.ten_velocity,
            "distance_from_origin": jnp.linalg.norm(data.qpos[0:2], ord=2),
            "reward_linup": uph_cost,
            "reward_quadctrl": -ctrl_cost,
            "reward_impact": -impact_cost,
        }

    def impact_cost(self, data: mjx.Data) -> Float[Array, ""]:
        raw_cost = self.impact_cost_weight * jnp.sum(jnp.square(data.cfrc_ext))
        min_cost, max_cost = self.impact_cost_range
        return jnp.clip(raw_cost, min_cost, max_cost)
