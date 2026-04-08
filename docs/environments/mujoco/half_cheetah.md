---
title: Half Cheetah
description: A planar two-legged runner that learns to sprint forward.
---

# Half Cheetah

Lerax port of Gymnasium's [Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) environment. A planar biped cat-like robot with 6 actuated joints runs along the +x axis. There is no notion of "falling"; the agent simply maximizes forward velocity while paying a small control cost.

## Observation space

`concat(qpos, qvel)` flattened. When `exclude_current_positions_from_observation=True` (default) `qpos[0]` (root x) is dropped. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 6 continuous joint torques.

## Reward

`forward_reward - ctrl_cost` where
`forward_reward = forward_reward_weight * (qpos[0]_next - qpos[0]) / dt` and
`ctrl_cost = ctrl_cost_weight * sum(action ** 2)` (default weights `1.0` and `0.1`).

## Termination

Never terminates. No built-in truncation.

::: lerax.env.mujoco.HalfCheetah
    options:
        members: false
