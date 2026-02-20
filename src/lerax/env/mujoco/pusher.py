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


class Pusher(AbstractMujocoEnv[Float[Array, "..."], Float[Array, "..."]]):
    """MJX-based pusher environment matching Gymnasium's Pusher-v5."""

    name: ClassVar[str] = "Pusher"

    action_space: Box
    observation_space: Box
    model: mjx.Model
    mujoco_model: mujoco.MjModel

    frame_skip: int
    dt: Float[Array, ""]

    reward_near_weight: Float[Array, ""]
    reward_dist_weight: Float[Array, ""]
    reward_control_weight: Float[Array, ""]

    tips_arm_body_id: int
    object_body_id: int
    goal_body_id: int

    init_qpos: Float[Array, "..."]
    init_qvel: Float[Array, "..."]

    def __init__(
        self,
        *,
        xml_file: str | Path = "pusher_v5.xml",
        frame_skip: int = 5,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1.0,
        reward_control_weight: float = 0.1,
    ):
        asset_path = Path(__file__).resolve().parent / "assets" / xml_file
        if not asset_path.exists():
            raise FileNotFoundError(f"Pusher asset not found: {asset_path}")

        mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        mj_data = mujoco.MjData(mj_model)

        self.model = mjx.put_model(mj_model)
        self.mujoco_model = mj_model

        self.frame_skip = int(frame_skip)
        self.dt = jnp.array(mj_model.opt.timestep * self.frame_skip)

        self.init_qpos = jnp.asarray(mj_data.qpos).reshape(-1)
        self.init_qvel = jnp.asarray(mj_data.qvel).reshape(-1)

        self.reward_near_weight = jnp.array(reward_near_weight)
        self.reward_dist_weight = jnp.array(reward_dist_weight)
        self.reward_control_weight = jnp.array(reward_control_weight)

        self.tips_arm_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "tips_arm"
        )
        self.object_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        self.goal_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

        # obs: qpos[:7] + qvel[:7] + tips_arm(3) + object(3) + goal(3) = 23
        obs_size = 23

        bounds = jnp.array(mj_model.actuator_ctrlrange, dtype=jnp.float32)
        low, high = bounds[:, 0], bounds[:, 1]
        self.action_space = Box(low=low, high=high)

        high_obs = jnp.full((obs_size,), jnp.inf, dtype=jnp.float32)
        self.observation_space = Box(low=-high_obs, high=high_obs)

    def initial(self, *, key: Key[Array, ""]) -> MujocoEnvState:
        cylinder_key, qvel_key = jr.split(key)

        qpos = self.init_qpos.copy()

        goal_pos = jnp.zeros(2)

        def cond_fn(state: tuple[Key[Array, ""], Float[Array, "2"]]) -> Bool[Array, ""]:
            _, pos = state
            return jnp.linalg.norm(pos - goal_pos) <= 0.17

        def body_fn(
            state: tuple[Key[Array, ""], Float[Array, "2"]],
        ) -> tuple[Key[Array, ""], Float[Array, "2"]]:
            rng, _ = state
            rng, x_key, y_key = jr.split(rng, 3)
            x = jr.uniform(x_key, minval=-0.3, maxval=0.0)
            y = jr.uniform(y_key, minval=-0.2, maxval=0.2)
            return rng, jnp.array([x, y])

        init_state = (cylinder_key, jnp.zeros(2))
        _, cylinder_pos = lax.while_loop(cond_fn, body_fn, init_state)

        qpos = qpos.at[-4:-2].set(cylinder_pos)
        qpos = qpos.at[-2:].set(goal_pos)

        qvel = self.init_qvel + jr.uniform(
            qvel_key, shape=self.init_qvel.shape, minval=-0.005, maxval=0.005
        )
        qvel = qvel.at[-4:].set(0.0)

        data = mjx.make_data(self.model)
        data = data.replace(qpos=qpos, qvel=qvel)

        return MujocoEnvState(sim_state=data, t=jnp.array(0.0))

    def observation(
        self, state: MujocoEnvState, *, key: Key[Array, ""]
    ) -> Float[Array, "..."]:
        data = state.sim_state

        tips_arm = data.xipos[self.tips_arm_body_id]
        obj = data.xipos[self.object_body_id]
        goal = data.xipos[self.goal_body_id]

        return jnp.concatenate(
            (
                data.qpos.reshape(-1)[:7],
                data.qvel.reshape(-1)[:7],
                tips_arm,
                obj,
                goal,
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

        tips_arm = data.xipos[self.tips_arm_body_id]
        obj = data.xipos[self.object_body_id]
        goal = data.xipos[self.goal_body_id]

        vec_near = obj - tips_arm
        vec_dist = obj - goal

        reward_near = -jnp.linalg.norm(vec_near) * self.reward_near_weight
        reward_dist = -jnp.linalg.norm(vec_dist) * self.reward_dist_weight
        reward_ctrl = -jnp.sum(jnp.square(action)) * self.reward_control_weight

        return reward_dist + reward_ctrl + reward_near

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

        tips_arm = data.xipos[self.tips_arm_body_id]
        obj = data.xipos[self.object_body_id]
        goal = data.xipos[self.goal_body_id]

        vec_near = obj - tips_arm
        vec_dist = obj - goal

        return {
            "reward_dist": -jnp.linalg.norm(vec_dist) * self.reward_dist_weight,
            "reward_ctrl": -jnp.sum(jnp.square(action)) * self.reward_control_weight,
            "reward_near": -jnp.linalg.norm(vec_near) * self.reward_near_weight,
        }
