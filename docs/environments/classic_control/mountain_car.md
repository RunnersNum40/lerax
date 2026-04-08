---
title: Mountain Car
description: Drive an underpowered car up a hill by rocking back and forth.
---

# Mountain Car

Lerax port of Gymnasium's [Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment. An underpowered car sits in a 1D valley and must reach a flag at the top of the right hill. The car cannot climb directly and must build momentum by swinging back and forth in the valley.

## Observation space

2-dim float vector `[position, velocity]`. Position is clipped to `[-1.2, 0.6]` and velocity to `[-0.07, 0.07]`.

## Action space

Discrete(3): `0` push left, `1` no push, `2` push right. Force applied is `(action - 1) * force` with `force = 0.001` by default.

## Reward

`-1.0` on every step regardless of action or outcome.

## Termination

Terminates when `position >= goal_position` (default `0.5`) and `velocity >= goal_velocity` (default `0.0`). No built-in truncation.

## Deviations from Gymnasium

Dynamics are integrated with Diffrax (default `Tsit5`); pass `solver=diffrax.Euler()` for Gymnasium-identical dynamics. No built-in 200-step truncation.

::: lerax.env.classic_control.MountainCar
    options:
        members: false
