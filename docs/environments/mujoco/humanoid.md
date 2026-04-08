---
title: Humanoid
description: A 3D bipedal humanoid that learns to walk forward.
---

# Humanoid

Lerax port of Gymnasium's [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) environment. A 3D humanoid with 17 actuated joints must walk in the +x direction while remaining upright.

## Observation space

Flattened concatenation of `qpos`, `qvel`, `cinert[1:]`, `cvel[1:]`, `qfrc_actuator[6:]`, and `cfrc_ext[1:]`. Each of the last four blocks is individually switchable via `include_*_in_observation` (all default `True`). When `exclude_current_positions_from_observation=True` (default) the first two entries of `qpos` are dropped. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 17 continuous joint torques.

## Reward

`forward_reward + healthy_reward - ctrl_cost - contact_cost` with

- `forward_reward = forward_reward_weight * x_velocity_of_center_of_mass` (default weight `1.25`)
- `healthy_reward = is_healthy * healthy_reward` (default `5.0`)
- `ctrl_cost = ctrl_cost_weight * sum(action ** 2)` (default `0.1`)
- `contact_cost = contact_cost_weight * clip(sum(cfrc_ext ** 2), -inf, 10.0)` (default weight `5e-7`)

## Termination

If `terminate_when_unhealthy=True` (default), terminates when torso z leaves `healthy_z_range=(1.0, 2.0)`. No built-in truncation.

::: lerax.env.mujoco.Humanoid
    options:
        members: false
