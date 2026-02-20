---
description: Unitree G1 Humanoid Robot Environments
---

# G1

The Unitree G1 is a 29-DOF humanoid robot with articulated legs (12 DOF), waist (3 DOF), and arms (14 DOF).
These environments use an MJX-optimized model with simplified collision geometry for efficient GPU/TPU simulation.

All G1 environments support:

- **Joint position control** with configurable action scaling
- **Configurable control frequency** (default 50 Hz)
- **Per-episode domain randomization** of friction, armature, masses, and joint friction loss
- **Observation noise** for sim-to-real robustness
- **Random push perturbations** for balance training

## Tasks

| Environment | Description |
|---|---|
| [G1Locomotion](locomotion.md) | Track velocity commands with a natural gait |
| [G1Standing](standing.md) | Maintain upright balance at default pose |
| [G1Standup](standup.md) | Stand up from a crouched position |
