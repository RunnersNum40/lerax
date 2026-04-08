---
title: Walker2d
description: A planar bipedal walker that learns to walk forward.
---

# Walker2d

Lerax port of Gymnasium's [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) environment. A planar biped (torso + two legs, 6 actuated joints) must walk along the +x axis without falling.

## Observation space

`concat(qpos, clip(qvel, -10, 10))` flattened. When `exclude_current_positions_from_observation=True` (default) `qpos[0]` is dropped. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 6 continuous joint torques.

## Reward

`forward_reward + healthy_reward - ctrl_cost` with
`forward_reward = forward_reward_weight * x_velocity`,
`healthy_reward = is_healthy * healthy_reward` (default `1.0`), and
`ctrl_cost = ctrl_cost_weight * sum(action ** 2)` (default `1e-3`).

## Termination

If `terminate_when_unhealthy=True` (default), terminates when torso z leaves `healthy_z_range=(0.8, 2.0)` or torso angle leaves `healthy_angle_range=(-1.0, 1.0)`. No built-in truncation.

::: lerax.env.mujoco.Walker2d
    options:
        members: false
