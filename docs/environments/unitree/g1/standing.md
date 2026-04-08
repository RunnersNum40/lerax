---
title: G1Standing
description: Keep the Unitree G1 humanoid upright at its default pose.
---

# G1Standing

The 29-DoF Unitree G1 must stand still at the default `knees_bent` pose under per-episode domain randomization and random push perturbations. Velocity commands are fixed to zero.

## Observation space

99-dim float vector, all with observation noise applied:

- local linear velocity (3)
- pelvis gyro (3)
- gravity vector in pelvis frame (3)
- velocity command, always `[0, 0, 0]` (3)
- joint angles relative to default pose (29)
- joint velocities (29)
- last action (29)

## Action space

`Box(-1, 1, shape=(29,))`. Scaled to joint position targets around the default pose: `targets = default_joint_positions + action * action_scale` (default `action_scale = 0.5`).

## Reward

Weighted sum, multiplied by `dt`:

- `orientation_weight * (1 - ||torso_gravity_xy|| ** 2)` (default `2.0`)
- `joint_vel_weight * sum(qvel[6:] ** 2)` (default `-0.1`)
- `pose_weight * sum((qpos[7:] - default) ** 2)` (default `-0.5`)
- `alive_weight` (default `1.0`)
- `action_rate_weight * sum((action - last_action) ** 2)` (default `-0.01`)
- `termination_weight * fallen` (default `-100.0`)

## Termination

Terminates on fall (`torso_gravity_z < 0`), NaN in `qpos`/`qvel`, or dangerous self-contacts (foot-foot or foot-shin). No built-in truncation.

## Deviations from Gymnasium

Not a Gymnasium environment. Uses MJX with per-episode domain randomization of friction, armature, body masses, joint friction loss, and torso inertia offset; applies observation noise and periodic random horizontal pushes.

::: lerax.env.unitree.g1.G1Standing
    options:
        members: false
