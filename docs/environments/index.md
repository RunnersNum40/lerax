---
title: Environments
---

# Environments

Lerax comes with several commonly used environments for reinforcement learning tasks.
All environments follow the [Lerax Environment API](../api/env.md) and support JAX transformations such as `jax.jit` and `jax.vmap`.

The Lerax environment API is a functional API, meaning that environments are immutable and all state changes return new environment states.
The logic for all parts of the reinforcement learning task are seperated into different components, such as the `inital`, `transition`, `observation`, `reward`, `terminal`, and `truncate` functions.
While this is a bit more verbose than traditional object-oriented APIs, it allows for greater flexibility and composability when implementing complex learning algorithms and simplifies JAX transformations.

For those that prefer the Gymnasium-style API `reset` and `step` functions are also provided as convenience methods that internally call the functional components.
These methods still follow the functional paradigm by returning new environment states rather than modifying the existing state in place.
If you'd like an identical Gymnasium-style API, you can use the [GymWrapper]() to wrap any Lerax environment.
