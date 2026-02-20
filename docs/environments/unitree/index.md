---
description: Unitree Robot Environments in the Lerax Reinforcement Learning Library
---

# Unitree

Environments for [Unitree](https://www.unitree.com/) humanoid robots using [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html), the JAX backend for MuJoCo.
These environments are designed for sim-to-real transfer with per-episode domain randomization, configurable control frequencies, and observation noise.

Robot models are adapted from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) with MJX optimizations from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

## Robots

| Robot | DOFs | Description |
|---|---|---|
| [G1](g1/index.md) | 29 | Full-body humanoid with legs, waist, and arms |
