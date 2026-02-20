---
description: MuJoCo Environments in the Lerax Reinforcement Learning Library
---

# MuJoCo

Lerax versions of the [Gymnasium MuJoCo Environments](https://gymnasium.farama.org/environments/mujoco/).
These environments use [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html), the JAX backend for MuJoCo, enabling fully JIT-compilable physics simulation on GPU/TPU accelerators.

All environments follow the Gymnasium v5 semantics and accept the same configuration parameters as their Gymnasium counterparts.

## Locomotion

| Environment | Description |
|---|---|
| [Ant](ant.md) | A four-legged creature that learns to walk forward |
| [HalfCheetah](half_cheetah.md) | A two-legged creature that learns to run |
| [Hopper](hopper.md) | A one-legged creature that learns to hop forward |
| [Humanoid](humanoid.md) | A bipedal robot that learns to walk |
| [Swimmer](swimmer.md) | A multi-jointed swimmer in a viscous fluid |
| [Walker2d](walker2d.md) | A bipedal walker that learns to walk forward |

## Standing

| Environment | Description |
|---|---|
| [HumanoidStandup](humanoid_standup.md) | A humanoid that learns to stand up from the ground |

## Balance

| Environment | Description |
|---|---|
| [InvertedPendulum](inverted_pendulum.md) | Balance a pole on a cart |
| [InvertedDoublePendulum](inverted_double_pendulum.md) | Balance two linked poles on a cart |

## Manipulation

| Environment | Description |
|---|---|
| [Pusher](pusher.md) | A robotic arm that pushes an object to a goal |
| [Reacher](reacher.md) | A robotic arm that reaches a target position |
