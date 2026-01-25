---
title: Getting Started
description: Documentation for Lerax, Jax based reinforcement learning library.
---

# Getting Started with Lerax

Do you want to leverage the power of JAX for high performance reinforcement learning?
Lerax is a reinforcement learning library built on top of JAX, designed to make it easy to implement and experiment with RL algorithms while taking advantage of JAX's speed and scalability.

Lerax provides **environments**, **policies**, and **training algorithms**. All with a modular design that makes it easy to compose different components together.

!!! warning "Work in Progress"

    Lerax is very much a work in progress, but it is already usable for training simple RL agents.
    The API is still evolving, and there are many features that are yet to be implemented.
    Additionally, the documentation is still being written, so please bear with me as I continue to improve it.

## Installation

```bash
pip install lerax
```

## Train a policy

```py title="examples/ppo.py"
--8<-- "examples/ppo.py"
```

## Acknowledgements

A ton of the code is a slight translation of the code found in the [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) libraries.
The developers of these excellent libraries have done a great job of creating a solid foundation for reinforcement learning in Python, and I have learned a lot from their design decisions.

In addition, the NDE code is heavily inspired by the work of [Patrick Kidger](https://kidger.site/publications/) and the entire library is based on his excellent [Equinox library](https://github.com/patrick-kidger/equinox) along with some use of [Diffrax](https://github.com/patrick-kidger/diffrax) and [jaxtyping](https://github.com/patrick-kidger/jaxtyping).
