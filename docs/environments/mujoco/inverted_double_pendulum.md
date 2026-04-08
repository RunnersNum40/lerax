---
title: Inverted Double Pendulum
description: Balance a two-link pole stack on a sliding cart.
---

# Inverted Double Pendulum

Lerax port of Gymnasium's [Inverted Double Pendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) environment. A MuJoCo cart with two un-actuated hinge links stacked on top. The agent applies horizontal cart force and must keep the tip near the upright position.

## Observation space

9-dim float vector: `[cart_x, sin(theta1), sin(theta2), cos(theta1), cos(theta2), clip(qvel, -10, 10)[0..2], clip(qfrc_constraint, -10, 10)[0]]`. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — a single continuous force applied to the cart.

## Reward

`alive_bonus - dist_penalty - vel_penalty` with

- `alive_bonus = (tip_y > 1) * healthy_reward` (default `healthy_reward = 10.0`)
- `dist_penalty = 0.01 * tip_x ** 2 + (tip_y - 2) ** 2`
- `vel_penalty = 1e-3 * qvel[1] ** 2 + 5e-3 * qvel[2] ** 2`

`tip_x`, `tip_y` are the x and z of `site_xpos[0]`.

## Termination

Terminates when the tip site z drops to `<= 1`. No built-in truncation.

::: lerax.env.mujoco.InvertedDoublePendulum
    options:
        members: false
