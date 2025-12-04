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

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ProgressBarCallback, TensorBoardCallback
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole() # (1)!
policy = MLPActorCriticPolicy(env=env, key=policy_key) # (2)!
algo = PPO() # (3)!
callbacks = [ProgressBarCallback(2**16), TensorBoardCallback(env=env, policy=policy)] # (4)!

policy = algo.learn( # (5)!
    env, policy, total_timesteps=2**16, key=learn_key, callback=callbacks
)
```

1. Create the environment. Lerax includes environments and wrappers for several popular RL environments.
2. Create the policy. Lerax includes several policy architectures and utilities to create custom policies.
3. Create the training algorithm. Lerax includes several algorithms and utilities to create custom algorithms.
4. Use callbacks to monitor training progress and log metrics.
5. Train the policy using the specified algorithm, environment, and callbacks.

## Acknowledgements

A ton of the code is a slight translation of the code found in the [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) libraries.
The developers of these excellent libraries have done a great job of creating a solid foundation for reinforcement learning in Python, and I have learned a lot from their design decisions.

In addition, the NDE code is heavily inspired by the work of [Patrick Kidger](https://kidger.site/publications/) and the entire library is based on his excellent [Equinox library](https://github.com/patrick-kidger/equinox) along with some use of [Diffrax](https://github.com/patrick-kidger/diffrax) and [jaxtyping](https://github.com/patrick-kidger/jaxtyping).
