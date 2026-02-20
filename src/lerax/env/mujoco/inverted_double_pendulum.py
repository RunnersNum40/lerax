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


class InvertedDoublePendulum(
    AbstractMujocoEnv[Float[Array, "..."], Float[Array, "..."]]
):
    """MJX-based inverted double pendulum environment matching Gymnasium's InvertedDoublePendulum-v5."""

    name: ClassVar[str] = "InvertedDoublePendulum"

    action_space: Box
    observation_space: Box
    model: mjx.Model
    mujoco_model: mujoco.MjModel

    frame_skip: int
    dt: Float[Array, ""]

    healthy_reward: Float[Array, ""]
    reset_noise_scale: Float[Array, ""]

    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    def __init__(
        self,
        *,
        xml_file: str | Path = "inverted_double_pendulum.xml",
        frame_skip: int = 5,
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.1,
    ):
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(
                f"InvertedDoublePendulum asset not found: {asset_path}"
            )

        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        mj_data = mujoco.MjData(mj_model)

        self.model = mjx.put_model(mj_model)
        self.mujoco_model = mj_model

        self.frame_skip = int(frame_skip)
        self.dt = jnp.array(mj_model.opt.timestep * self.frame_skip)

        self.init_qpos = jnp.asarray(mj_data.qpos).reshape(-1)
        self.init_qvel = jnp.asarray(mj_data.qvel).reshape(-1)

        self.healthy_reward = jnp.array(healthy_reward)
        self.reset_noise_scale = jnp.array(reset_noise_scale)

        # Observation: qpos[:1] + sin(qpos[1:]) + cos(qpos[1:]) + clip(qvel) + clip(qfrc_constraint)[:1]
        # = 1 + 2 + 2 + 3 + 1 = 9
        obs_size = 9

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
        qpos = data.qpos.reshape(-1)
        qvel = data.qvel.reshape(-1)

        return jnp.concatenate(
            (
                qpos[:1],
                jnp.sin(qpos[1:]),
                jnp.cos(qpos[1:]),
                jnp.clip(qvel, -10.0, 10.0),
                jnp.clip(data.qfrc_constraint.reshape(-1), -10.0, 10.0)[:1],
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

        site_pos = data.site_xpos[0]
        x = site_pos[0]
        y = site_pos[2]

        v1, v2 = data.qvel[1], data.qvel[2]

        alive = (y > 1).astype(float) * self.healthy_reward
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2

        return alive - dist_penalty - vel_penalty

    def terminal(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        y = state.sim_state.site_xpos[0][2]
        return y <= 1

    def transition_info(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
    ) -> dict:
        data = next_state.sim_state
        site_pos = data.site_xpos[0]
        x = site_pos[0]
        y = site_pos[2]
        v1, v2 = data.qvel[1], data.qvel[2]

        return {
            "dist_penalty": 0.01 * x**2 + (y - 2) ** 2,
            "vel_penalty": 1e-3 * v1**2 + 5e-3 * v2**2,
            "alive_bonus": (y > 1).astype(float) * self.healthy_reward,
        }
