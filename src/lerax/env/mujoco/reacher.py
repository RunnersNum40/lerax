from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import mujoco
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key
from mujoco import mjx

from lerax.space import Box

from .base_mujoco import AbstractMujocoEnv, MujocoEnvState


class Reacher(AbstractMujocoEnv[Float[Array, "..."], Float[Array, "..."]]):
    """MJX-based reacher environment matching Gymnasium's Reacher-v5."""

    name: ClassVar[str] = "Reacher"

    action_space: Box
    observation_space: Box
    model: mjx.Model
    mujoco_model: mujoco.MjModel

    frame_skip: int
    dt: Float[Array, ""]

    reward_dist_weight: Float[Array, ""]
    reward_control_weight: Float[Array, ""]

    fingertip_body_id: int
    target_body_id: int

    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    def __init__(
        self,
        *,
        xml_file: str | Path = "reacher.xml",
        frame_skip: int = 2,
        reward_dist_weight: float = 1.0,
        reward_control_weight: float = 1.0,
    ):
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(f"Reacher asset not found: {asset_path}")

        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        mj_data = mujoco.MjData(mj_model)

        self.model = mjx.put_model(mj_model)
        self.mujoco_model = mj_model

        self.frame_skip = int(frame_skip)
        self.dt = jnp.array(mj_model.opt.timestep * self.frame_skip)

        self.init_qpos = jnp.asarray(mj_data.qpos).reshape(-1)
        self.init_qvel = jnp.asarray(mj_data.qvel).reshape(-1)

        self.reward_dist_weight = jnp.array(reward_dist_weight)
        self.reward_control_weight = jnp.array(reward_control_weight)

        self.fingertip_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "fingertip"
        )
        self.target_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        # obs: cos(theta[:2]) + sin(theta[:2]) + target(2) + omega[:2] + diff[:2] = 10
        obs_size = 10

        bounds = jnp.array(mj_model.actuator_ctrlrange, dtype=jnp.float32)
        low, high = bounds[:, 0], bounds[:, 1]
        self.action_space = Box(low=low, high=high)

        high_obs = jnp.full((obs_size,), jnp.inf, dtype=jnp.float32)
        self.observation_space = Box(low=-high_obs, high=high_obs)

    def initial(self, *, key: Key[Array, ""]) -> MujocoEnvState:
        qpos_key, goal_key, qvel_key = jr.split(key, 3)

        qpos = self.init_qpos + jr.uniform(
            qpos_key, shape=self.init_qpos.shape, minval=-0.1, maxval=0.1
        )

        def cond_fn(state: tuple[Key[Array, ""], Float[Array, "2"]]) -> Bool[Array, ""]:
            _, goal = state
            return jnp.linalg.norm(goal) >= 0.2

        def body_fn(
            state: tuple[Key[Array, ""], Float[Array, "2"]],
        ) -> tuple[Key[Array, ""], Float[Array, "2"]]:
            rng, _ = state
            rng, goal_sub_key = jr.split(rng)
            goal = jr.uniform(goal_sub_key, shape=(2,), minval=-0.2, maxval=0.2)
            return rng, goal

        init_state = (goal_key, jnp.full((2,), 0.2))
        _, goal = lax.while_loop(cond_fn, body_fn, init_state)

        qpos = qpos.at[-2:].set(goal)

        qvel = self.init_qvel + jr.uniform(
            qvel_key, shape=self.init_qvel.shape, minval=-0.005, maxval=0.005
        )
        qvel = qvel.at[-2:].set(0.0)

        data = mjx.make_data(self.model)
        data = data.replace(qpos=qpos, qvel=qvel)

        return MujocoEnvState(sim_state=data, t=jnp.array(0.0))

    def observation(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Float[Array, "..."]:
        data = state.sim_state

        theta = data.qpos.reshape(-1)[:2]
        fingertip = data.xipos[self.fingertip_body_id]
        target = data.xipos[self.target_body_id]

        return jnp.concatenate(
            (
                jnp.cos(theta),
                jnp.sin(theta),
                data.qpos.reshape(-1)[2:],
                data.qvel.reshape(-1)[:2],
                (fingertip - target)[:2],
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

        fingertip = data.xipos[self.fingertip_body_id]
        target = data.xipos[self.target_body_id]
        vec = fingertip - target

        reward_dist = -jnp.linalg.norm(vec) * self.reward_dist_weight
        reward_ctrl = -jnp.sum(jnp.square(action)) * self.reward_control_weight

        return reward_dist + reward_ctrl

    def terminal(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return jnp.array(False)

    def transition_info(
        self,
        state: MujocoEnvState,
        action: Float[Array, "..."],
        next_state: MujocoEnvState,
    ) -> dict:
        data = next_state.sim_state

        fingertip = data.xipos[self.fingertip_body_id]
        target = data.xipos[self.target_body_id]
        vec = fingertip - target

        return {
            "reward_dist": -jnp.linalg.norm(vec) * self.reward_dist_weight,
            "reward_ctrl": -jnp.sum(jnp.square(action)) * self.reward_control_weight,
        }
