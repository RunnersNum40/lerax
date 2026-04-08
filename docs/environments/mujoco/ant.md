---
title: Ant
description: A four-legged MJX creature that learns to walk forward.
---

# Ant

Lerax port of Gymnasium's [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) environment. A quadruped (torso + 4 two-link legs, 8 actuated hinges) is dropped on flat ground and must run in the +x direction without flipping over.

## Observation space

Concatenation of `qpos`, `qvel`, and clipped external contact forces `cfrc_ext[1:]` flattened. By default `exclude_current_positions_from_observation=True` drops `qpos[:2]` (root x/y) and `include_cfrc_ext_in_observation=True` includes contact forces. Observation is unbounded (`Box(-inf, inf)`).

## Action space

`Box(low, high)` taken from the MJX model's `actuator_ctrlrange` — 8 continuous torques, one per hinge.

## Reward

`forward_reward + healthy_reward - ctrl_cost - contact_cost` where:

- `forward_reward = forward_reward_weight * x_velocity` (x velocity computed from torso `xpos` change over `dt`)
- `healthy_reward = is_healthy * healthy_reward` (default `1.0`)
- `ctrl_cost = ctrl_cost_weight * sum(action ** 2)` (default weight `0.5`)
- `contact_cost = contact_cost_weight * sum(clip(cfrc_ext, -1, 1) ** 2)` (default weight `5e-4`)

## Termination

Terminates when unhealthy if `terminate_when_unhealthy=True` (default). Healthy means `qpos`/`qvel` are finite and torso z is within `healthy_z_range=(0.2, 1.0)`. No built-in truncation.

::: lerax.env.mujoco.Ant
    options:
        members: false
