---
title: Getting Started
description: Documentation for Lerax, Jax based reinforcement learning library.
---

## Installation

```bash
pip install lerax
```

## Train a policy

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()

policy = algo.learn(env, policy, total_timesteps=2**16, key=learn_key)
```

## Core Concepts

### Functional Environments

Environments are follow in a functional style that is more verbose than the traditional OpenAI Gym interface.
In addition they do not maintain any internal state, instead the state is passed around explicitly.

They expose the following methods[^1]:
[^1]: This doc style will be changed when the API documentation is added.

- `initial`: Return an initial state.
- `transition`: Given a state and action, return the next state.
- `reward`: Given a state, action, and next state, return the reward.
- `terminal`: Given a state, return whether it is terminal.
- `truncate`: Given a state, return whether it is truncated.
- `state_info`: Given a state, return any additional information about the state.
- `transition_info`: Given a state, action, and next state, return any additional information about the transition.

### Custom Policies

Although Lerax provides some out of the box policies, users are encouraged to implement their own policies.
See [the custom policy guide](./examples/custom_policy) for more information.

## Acknowledgements

A ton of the code is a slight translation of the code found in the [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) libraries.
The developers of these excellent libraries have done a great job of creating a solid foundation for reinforcement learning in Python, and I have learned a lot from their design decisions.

In addition, the NDE code is heavily inspired by the work of [Patrick Kidger](https://kidger.site/publications/) and the entire library is based on his excellent [Equinox library](https://github.com/patrick-kidger/equinox) along with some use of [Diffrax](https://github.com/patrick-kidger/diffrax) and [jaxtyping](https://github.com/patrick-kidger/jaxtyping).
