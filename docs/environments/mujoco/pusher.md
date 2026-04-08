---
title: Pusher
description: A 7-DoF arm that pushes a cylinder to a target location.
---

# Pusher

Lerax port of Gymnasium's [Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/) environment. A 7-DoF arm on a fixed base must push a free cylinder lying on the table to a fixed goal position. The cylinder is re-spawned each episode to a random position at least `0.17` from the goal.

## Observation space

23-dim float vector: `concat(qpos[:7], qvel[:7], tips_arm_xipos(3), object_xipos(3), goal_xipos(3))`. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 7 continuous joint torques.

## Reward

`reward_dist + reward_ctrl + reward_near` where

- `reward_near = -||object - tips_arm|| * reward_near_weight` (default `0.5`)
- `reward_dist = -||object - goal|| * reward_dist_weight` (default `1.0`)
- `reward_ctrl = -sum(action ** 2) * reward_control_weight` (default `0.1`)

## Termination

Never terminates. No built-in truncation.

::: lerax.env.mujoco.Pusher
    options:
        members: false
