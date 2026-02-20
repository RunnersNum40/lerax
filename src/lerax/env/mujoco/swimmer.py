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


class Swimmer(AbstractMujocoEnv[Float[Array, "..."], Float[Array, "..."]]):
    """MJX-based swimmer environment matching Gymnasium's Swimmer-v5."""

    name: ClassVar[str] = "Swimmer"

    action_space: Box
    observation_space: Box
    model: mjx.Model
    mujoco_model: mujoco.MjModel

    frame_skip: int
    dt: Float[Array, ""]

    forward_reward_weight: Float[Array, ""]
    ctrl_cost_weight: Float[Array, ""]

    reset_noise_scale: Float[Array, ""]
    exclude_current_positions_from_observation: bool

    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    def __init__(
        self,
        *,
        xml_file: str | Path = "swimmer.xml",
        frame_skip: int = 4,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
    ):
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(f"Swimmer asset not found: {asset_path}")

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

        self.reset_noise_scale = jnp.array(reset_noise_scale)
        self.exclude_current_positions_from_observation = bool(
            exclude_current_positions_from_observation
        )

        skipped_qpos = 2 if self.exclude_current_positions_from_observation else 0
        obs_size = mj_data.qpos.size + mj_data.qvel.size - skipped_qpos

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

        return jnp.concatenate((position, velocity))

    def reward(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        x_before = state.sim_state.qpos[0]
        x_after = next_state.sim_state.qpos[0]
        x_velocity = (x_after - x_before) / self.dt

        forward_reward = self.forward_reward_weight * x_velocity
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))

        return forward_reward - ctrl_cost

    def terminal(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return jnp.array(False)

    def state_info(self, state: MujocoEnvState) -> dict:
        return {
            "x_position": state.sim_state.qpos[0],
            "y_position": state.sim_state.qpos[1],
            "distance_from_origin": jnp.linalg.norm(state.sim_state.qpos[0:2], ord=2),
        }

    def transition_info(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
    ) -> dict:
        x_before = state.sim_state.qpos[0]
        x_after = next_state.sim_state.qpos[0]
        x_velocity = (x_after - x_before) / self.dt

        forward_reward = self.forward_reward_weight * x_velocity
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))

        return {
            "x_position": next_state.sim_state.qpos[0],
            "y_position": next_state.sim_state.qpos[1],
            "distance_from_origin": jnp.linalg.norm(
                next_state.sim_state.qpos[0:2], ord=2
            ),
            "x_velocity": x_velocity,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
