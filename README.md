# Lerax: Fully JITable reinforcement learning with Jax.

Lerax is a reinforcement learning library built on top of Jax, designed to facilitate the creation, training, and evaluation of RL agents in a fully JITable manner.
It provides modular components for building custom environments, policies, and training algorithms.

Built on top of [Jax](https://docs.jax.dev/en/latest/index.html), [Equinox](https://docs.kidger.site/equinox/), and [Diffrax](https://docs.kidger.site/diffrax/).

## Installation

```bash
pip install lerax
```

## Documentation

Check out: [lerax.tedpinkerton.ca](https://lerax.tedpinkerton.ca)

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

## TODO

- Optimise for performance under JIT compilation
  - Sharding support for distributed training
- Documentation
  - Standardize docstring formats
  - Write documentation for all public APIs
  - Add API to docs when Zensical supports it
- Testing
  - Unit testing
  - Integration testing
  - Full Jaxtyping
    - Ensure all functions and classes have proper type annotations
- Round out features
  - Expand RL variants to include more algorithms
    - Any off-policy algorithms
  - Create a more comprehensive set of environments
    - Brax based environments
