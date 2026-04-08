---
title: Swimmer
description: A multi-link swimmer that learns to move through a viscous fluid.
---

# Swimmer

Lerax port of Gymnasium's [Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/) environment. A planar three-link swimmer with two actuated hinges moves through a viscous medium. The agent maximizes x-velocity.

## Observation space

`concat(qpos, qvel)` flattened. When `exclude_current_positions_from_observation=True` (default) the first two entries of `qpos` (x, y) are dropped. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 2 continuous joint torques.

## Reward

`forward_reward - ctrl_cost` with
`forward_reward = forward_reward_weight * (qpos[0]_next - qpos[0]) / dt` and
`ctrl_cost = ctrl_cost_weight * sum(action ** 2)` (default weights `1.0` and `1e-4`).

## Termination

Never terminates. No built-in truncation.

::: lerax.env.mujoco.Swimmer
    options:
        members: false
