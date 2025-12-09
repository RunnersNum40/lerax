# Lerax: Fully JITable reinforcement learning with Jax.

Lerax is a reinforcement learning library built on top of Jax, designed to facilitate the creation, training, and evaluation of RL agents in a fully JITable manner.
It provides modular components for building custom environments, policies, and training algorithms.

Built on top of [Jax](https://docs.jax.dev/en/latest/index.html), [Equinox](https://docs.kidger.site/equinox/), and [Diffrax](https://docs.kidger.site/diffrax/).

## Installation

```bash
pip install lerax
```

## Training Example

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=jr.key(0))
algo = PPO()

policy = algo.learn(env, policy, total_timesteps=2**16, key=jr.key(1))
```

## Documentation

Check out: [lerax.tedpinkerton.ca](https://lerax.tedpinkerton.ca)
