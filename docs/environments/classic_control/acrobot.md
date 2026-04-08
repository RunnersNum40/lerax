---
title: Acrobot
description: Swing a two-link pendulum up so the tip rises above the base.
---

# Acrobot

Lerax port of Gymnasium's [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) environment. Two rigid links are connected in series, with the first link pinned to the base and only the joint between the two links actuated. The agent applies torque at the middle joint to swing the lower tip up above a target height.

## Observation space

6-dim float vector `[cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]`. Velocities are clipped to `±4π` and `±9π` respectively.

## Action space

Discrete(3) selecting a torque from `torques = [-1.0, 0.0, 1.0]` (applied to the joint between the two links).

## Reward

`-1.0` on every non-terminal step, `0.0` on the terminal step. Computed as `done.astype(float) - 1.0`.

## Termination

Terminates when the tip of the second link rises above `1.0`, i.e. `-cos(theta1) - cos(theta1 + theta2) > 1.0`. No built-in truncation.

## Deviations from Gymnasium

Dynamics are integrated with Diffrax (default `Tsit5`); pass `solver=diffrax.Euler()` for Gymnasium-identical dynamics. No built-in 500-step truncation.

::: lerax.env.classic_control.Acrobot
    options:
        members: false
