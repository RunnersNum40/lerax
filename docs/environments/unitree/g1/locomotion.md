---
title: G1Locomotion
description: Track velocity commands on the Unitree G1 with a phased bipedal gait.
---

# G1Locomotion

The 29-DoF Unitree G1 must track randomly sampled `[vx, vy, yaw_rate]` velocity commands while maintaining a periodic bipedal gait. A gait phase is carried on the state and advanced at a per-episode sampled frequency. Domain randomization, observation noise, and random pushes are applied for sim-to-real robustness.

## Observation space

103-dim float vector with observation noise:

- local linear velocity (3)
- pelvis gyro (3)
- gravity vector in pelvis frame (3)
- velocity command `[vx, vy, yaw_rate]` (3)
- joint angles relative to default pose (29)
- joint velocities (29)
- last action (29)
- gait phase `[cos_l, sin_l, cos_r, sin_r]` (4)

## Action space

`Box(-1, 1, shape=(29,))`. Scaled to joint position targets around the default pose: `targets = default_joint_positions + action * action_scale` (default `action_scale = 0.5`).

## Reward

Weighted sum multiplied by `dt`. Major terms (defaults in parentheses):

- `tracking_lin_vel` (`1.0`): `exp(-||cmd_xy - local_linvel_xy||^2 / tracking_sigma)`
- `tracking_ang_vel` (`0.75`): `exp(-(cmd_yaw - gyro_z)^2 / tracking_sigma)`
- `ang_vel_xy` (`-0.15`), `orientation` (`-2.0`), `feet_slip` (`-0.25`)
- `feet_air_time` (`2.0`) rewarding first ground contacts with air time in `[0.2, 0.5]`
- `feet_phase` (`1.0`): `exp(-phase_error / 0.01) * moving`, where `phase_error = sum((foot_z - desired_z(phase))^2)`
- `termination` (`-100.0`) on fall/self-contact, `stand_still` (`-1.0`) under zero command
- `collision` (`-0.1`) on hand-thigh contacts, `dof_pos_limits` (`-1.0`) outside soft limits
- `pose` (`-0.1`), `joint_deviation_hip` (`-0.25`), `joint_deviation_knee` (`-0.1`)
- `torques`, `action_rate`, `dof_acc`, `alive` default to `0.0`

All weights are overridable via the `reward_weights` dict.

## Termination

Terminates on fall (`torso_gravity_z < 0`), NaN in `qpos`/`qvel`, or dangerous self-contacts (foot-foot, foot-shin). No built-in truncation.

## Deviations from Gymnasium

Not a Gymnasium environment. Uses MJX with per-episode domain randomization (friction, armature, body masses, joint friction loss, torso inertia offset), observation noise, periodic random horizontal pushes, and a gait-phase signal carried on state. Velocity commands are resampled per episode; with `zero_command_probability=0.1` the command is zeroed to teach standing still.

::: lerax.env.unitree.g1.G1Locomotion
    options:
        members: false
