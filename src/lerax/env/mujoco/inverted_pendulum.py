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


class InvertedPendulum(AbstractMujocoEnv[Float[Array, "..."], Float[Array, "..."]]):
    """MJX-based inverted pendulum environment matching Gymnasium's InvertedPendulum-v5."""

    name: ClassVar[str] = "InvertedPendulum"

    action_space: Box
    observation_space: Box
    model: mjx.Model
    mujoco_model: mujoco.MjModel

    frame_skip: int
    dt: Float[Array, ""]

    reset_noise_scale: Float[Array, ""]

    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    def __init__(
        self,
        *,
        xml_file: str | Path = "inverted_pendulum.xml",
        frame_skip: int = 2,
        reset_noise_scale: float = 0.01,
    ):
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(f"InvertedPendulum asset not found: {asset_path}")

        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        mj_data = mujoco.MjData(mj_model)

        self.model = mjx.put_model(mj_model)
        self.mujoco_model = mj_model

        self.frame_skip = int(frame_skip)
        self.dt = jnp.array(mj_model.opt.timestep * self.frame_skip)

        self.init_qpos = jnp.asarray(mj_data.qpos).reshape(-1)
        self.init_qvel = jnp.asarray(mj_data.qvel).reshape(-1)

        self.reset_noise_scale = jnp.array(reset_noise_scale)

        obs_size = mj_data.qpos.size + mj_data.qvel.size

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
        return jnp.concatenate((data.qpos.reshape(-1), data.qvel.reshape(-1)))

    def reward(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        not_terminated = self.is_healthy(next_state.sim_state)
        return not_terminated.astype(float)

    def terminal(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return ~self.is_healthy(state.sim_state)

    def transition_info(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
    ) -> dict:
        return {}

    def is_healthy(self, data: mjx.Data) -> Bool[Array, ""]:
        obs = jnp.concatenate((data.qpos.reshape(-1), data.qvel.reshape(-1)))
        is_finite = jnp.all(jnp.isfinite(obs))
        angle_ok = jnp.abs(data.qpos[1]) <= 0.2
        return is_finite & angle_ok
