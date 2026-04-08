---
title: Cart Pole
description: Balance a pole on a moving cart by pushing left or right.
---

# Cart Pole

Lerax port of Gymnasium's [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. A pole is attached by an un-actuated joint to a cart that moves along a frictionless track. The agent applies a fixed-magnitude force to the cart at each step and must keep the pole upright and the cart near the center of the track for as long as possible.

## Observation space

4-dim float vector `[cart_x, cart_x_dot, pole_theta, pole_theta_dot]`. The `Box` bounds are set to twice the termination thresholds (`±2 * x_threshold` for `cart_x`, `±2 * theta_threshold_radians` for `pole_theta`) and `±inf` for the two velocities.

## Action space

Discrete(2). `0` pushes the cart left, `1` pushes it right. Internally the force is `(2 * action - 1) * force_mag` (default `force_mag = 10.0`).

## Reward

`+1.0` on every step, including the terminating step.

## Termination

Terminates when `|cart_x| > x_threshold` (default `2.4`) or `|pole_theta| > theta_threshold_radians` (default `12°`). There is no built-in truncation or time limit.

## Deviations from Gymnasium

Dynamics are integrated with Diffrax (default `Tsit5` with `ConstantStepSize`) rather than Euler. Pass `solver=diffrax.Euler()` to reproduce Gymnasium exactly. No built-in 500-step truncation; wrap the env if you want one.

::: lerax.env.classic_control.CartPole
    options:
        members: false
