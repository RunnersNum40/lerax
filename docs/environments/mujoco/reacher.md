---
title: Reacher
description: A 2-link planar arm that touches a randomly placed target.
---

# Reacher

Lerax port of Gymnasium's [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) environment. A planar two-joint arm must move its fingertip to a random target sampled within radius `0.2` of the base each episode.

## Observation space

10-dim float vector: `[cos(theta1), cos(theta2), sin(theta1), sin(theta2), target_x, target_y, theta1_dot, theta2_dot, (fingertip - target)_x, (fingertip - target)_y]`. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 2 continuous joint torques.

## Reward

`reward_dist + reward_ctrl` where

- `reward_dist = -||fingertip - target|| * reward_dist_weight` (default `1.0`)
- `reward_ctrl = -sum(action ** 2) * reward_control_weight` (default `1.0`)

## Termination

Never terminates. No built-in truncation.

::: lerax.env.mujoco.Reacher
    options:
        members: false
