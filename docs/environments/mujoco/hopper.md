---
title: Hopper
description: A one-legged planar robot that learns to hop forward.
---

# Hopper

Lerax port of Gymnasium's [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) environment. A 2D one-legged robot (torso + thigh + leg + foot, 3 actuated joints) must hop along the +x axis without falling over.

## Observation space

`concat(qpos, clip(qvel, -10, 10))` flattened. When `exclude_current_positions_from_observation=True` (default) `qpos[0]` is dropped. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 3 continuous joint torques.

## Reward

`forward_reward + healthy_reward - ctrl_cost` with
`forward_reward = forward_reward_weight * x_velocity`,
`healthy_reward = is_healthy * healthy_reward` (default `1.0`), and
`ctrl_cost = ctrl_cost_weight * sum(action ** 2)` (default `1e-3`).

## Termination

If `terminate_when_unhealthy=True` (default), terminates when not healthy. Healthy means every entry of `concat(qpos[2:], qvel)` lies in `healthy_state_range=(-100, 100)`, torso z in `healthy_z_range=(0.7, inf)`, and torso angle in `healthy_angle_range=(-0.2, 0.2)`. No built-in truncation.

::: lerax.env.mujoco.Hopper
    options:
        members: false
