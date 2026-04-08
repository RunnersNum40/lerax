---
title: G1Standup
description: Bring the Unitree G1 to full standing height from a low pose.
---

# G1Standup

The 29-DoF Unitree G1 starts from a perturbed, lowered pose (`keyframe.qpos` with `z` reduced by `0.1`, snapped to the ground). The agent is rewarded for raising its torso height. The episode never terminates early.

## Observation space

Same 99-dim layout as [G1Standing](standing.md): noisy linvel(3) + gyro(3) + gravity(3) + command(3, zeros) + joint angle offsets(29) + joint velocities(29) + last action(29).

## Action space

`Box(-1, 1, shape=(29,))`. Scaled to joint position targets around the default pose: `targets = default_joint_positions + action * action_scale` (default `action_scale = 0.5`).

## Reward

`height_weight * (torso_z / dt) - ctrl_cost_weight * sum(data.ctrl ** 2) + orientation_weight * (1 - ||torso_gravity_xy|| ** 2) + 1.0`

Defaults: `height_weight=1.0`, `ctrl_cost_weight=0.1`, `orientation_weight=0.5`. Note the height term uses `torso_z / dt` (analogous to Gymnasium's HumanoidStandup).

## Termination

Never terminates. No built-in truncation.

## Deviations from Gymnasium

Not a Gymnasium environment. Uses MJX with per-episode domain randomization (friction, armature, masses, torso offset). Pushes are disabled by default for this task.

::: lerax.env.unitree.g1.G1Standup
    options:
        members: false
