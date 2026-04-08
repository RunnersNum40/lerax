---
title: Continuous Mountain Car
description: Mountain Car with a continuous throttle and a terminal bonus.
---

# Continuous Mountain Car

Lerax port of Gymnasium's [Continuous Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) environment. Same physical setup as Mountain Car, but the agent applies a continuous throttle in `[-1, 1]` and is rewarded for reaching the flag while penalized for action magnitude.

## Observation space

2-dim float vector `[position, velocity]`. Position is clipped to `[-1.2, 0.6]` and velocity to `[-0.07, 0.07]`.

## Action space

`Box(low=-1.0, high=1.0, shape=())`. The acceleration contribution is `power * clip(action, -1, 1)` with `power = 0.0015`.

## Reward

`100 * terminal - 0.1 * clip(action, -1, 1) ** 2`: a large one-time bonus for reaching the goal plus a small quadratic action cost every step.

## Termination

Terminates when `position >= goal_position` (default `0.5`) and `velocity >= goal_velocity` (default `0.0`). No built-in truncation.

## Deviations from Gymnasium

Dynamics are integrated with Diffrax (default `Tsit5`); pass `solver=diffrax.Euler()` for Gymnasium-identical dynamics. No built-in 999-step truncation.

::: lerax.env.classic_control.ContinuousMountainCar
    options:
        members: false
