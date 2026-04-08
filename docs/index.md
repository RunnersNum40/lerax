---
title: Getting Started
description: A JAX-native reinforcement learning library.
---

!!! warning "Work in Progress"

    Lerax is usable for training simple RL agents, but the API is still
    evolving and the documentation is incomplete. Expect rough edges.

# Getting Started with Lerax

Lerax is a reinforcement learning library built on [JAX](https://docs.jax.dev/) and [Equinox](https://github.com/patrick-kidger/equinox).
It provides functional **environments**, **policies**, and **training algorithms** that compose cleanly under `jax.jit` and `jax.vmap`.

## Installation

```bash
pip install lerax
```

## Train a policy

The example below trains a PPO agent on CartPole and streams metrics to both the terminal and TensorBoard:

```py title="examples/ppo.py"
--8<-- "examples/ppo.py"
```

That's a complete training run — no separate config file, no custom training loop. The same shape works for any combination of environment, policy, and algorithm in the library.

## Next steps

- [Environments](environments/index.md) — built-in environments and how to write your own.
- [Compatibility](compatibility.md) — using Gymnasium, Gymnax, and Stable Baselines3 environments and algorithms with Lerax.
- [Callbacks](callbacks/index.md) — logging, progress bars, and custom training hooks.
- [Saving & Loading](saving_and_loading.md) — serializing policies and exporting to ONNX.

## Acknowledgements

A large amount of the code is a translation of patterns from [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium); both libraries are excellent foundations for RL in Python and Lerax owes a lot to their design.

The NDE code is heavily inspired by the work of [Patrick Kidger](https://kidger.site/publications/), and the entire library is built on his [Equinox](https://github.com/patrick-kidger/equinox), [Diffrax](https://github.com/patrick-kidger/diffrax), and [jaxtyping](https://github.com/patrick-kidger/jaxtyping) libraries.
