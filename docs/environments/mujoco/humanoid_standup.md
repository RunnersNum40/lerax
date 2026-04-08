---
title: Humanoid Standup
description: A humanoid that learns to stand up from a lying start pose.
---

# Humanoid Standup

Lerax port of Gymnasium's [Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) environment. Same humanoid model as [Humanoid](humanoid.md), but starts lying on the ground. The agent is rewarded for raising its torso z-coordinate as high as possible.

## Observation space

Same layout as Humanoid: flattened concatenation of `qpos`, `qvel`, `cinert[1:]`, `cvel[1:]`, `qfrc_actuator[6:]`, and `cfrc_ext[1:]`, each individually switchable. Current x/y are dropped by default. Unbounded `Box`.

## Action space

`Box(low, high)` from the model's `actuator_ctrlrange` — 17 continuous joint torques.

## Reward

`uph_cost - ctrl_cost - impact_cost + 1` where

- `uph_cost = uph_cost_weight * (qpos[2] / dt)` (torso z divided by control dt; default weight `1.0`)
- `ctrl_cost = ctrl_cost_weight * sum(data.ctrl ** 2)` (default `0.1`)
- `impact_cost = clip(impact_cost_weight * sum(cfrc_ext ** 2), -inf, 10.0)` (default weight `0.5e-6`)

## Termination

Never terminates. No built-in truncation.

::: lerax.env.mujoco.HumanoidStandup
    options:
        members: false
