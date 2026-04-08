---
title: Inverted Pendulum
description: Balance a single pole on a sliding cart using MuJoCo.
---

# Inverted Pendulum

Lerax port of Gymnasium's [Inverted Pendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) environment. A MuJoCo cart-pole: a sliding cart supports an un-actuated hinge pole. The agent applies horizontal force to the cart to keep the pole upright.

## Observation space

`concat(qpos, qvel)` flattened — a 4-dim unbounded float vector `[cart_x, pole_angle, cart_x_dot, pole_angle_dot]`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — a single continuous force applied to the cart.

## Reward

`+1.0` per step while healthy, `0.0` otherwise (`not_terminated.astype(float)`).

## Termination

Terminates when state becomes non-finite or `|pole_angle| > 0.2` rad. No built-in truncation.

::: lerax.env.mujoco.InvertedPendulum
    options:
        members: false
