---
title: Making a Custom Policy
description: Example of creating a custom policy in Lerax reinforcement learning library.
status: draft
---

While Lerax provides several built-in policies, you may want to create your own custom policy to better suit your specific task.

## Actor-Critic Policy

Actor-Critic policies combine both policy-based and value-based methods.
Below is an example of how to create a custom Actor-Critic policy by subclassing the `ActorCriticPolicy` class.

```py
from jax import numpy as jnp
from jax import random as jr

from lerax.distribution import Categorical
from lerax.env import AbstractEnvLike
from lerax.model import MLP
from lerax.policy import AbstractStatelessActorCriticPolicy
from lerax.space import Box, Discrete


class CustomActorCriticPolicy(AbstractStatelessActorCriticPolicy):
    action_space: Discrete
    observation_space: Discrete | Box

    action_head: MLP
    value_head: MLP

    name: str = "CustomActorCriticPolicy"

    def __init__(self, env: AbstractEnvLike, *, key):
        assert isinstance(env.action_space, Discrete)
        self.action_space = env.action_space
        assert isinstance(env.observation_space, (Discrete, Box))
        self.observation_space = env.observation_space

        act_key, val_key = jr.split(key, 2)
        self.action_head = MLP(
            in_size=int(jnp.array(self.observation_space.flat_size)),
            out_size=int(jnp.array(self.action_space.n)),
            width_size=32,
            depth=2,
            key=act_key,
        )

        self.value_head = MLP(
            in_size=int(jnp.array(self.observation_space.flat_size)),
            out_size="scalar",
            width_size=32,
            depth=2,
            key=val_key,
        )

    def features(self, observation):
        return self.observation_space.flatten_sample(observation)

    def action_dist(self, observation):
        return Categorical(self.action_head(self.features(observation)))

    def __call__(self, observation, *, key=None):
        if key is None:
            return self.action_dist(observation).mode()
        return self.action_dist(observation).sample(key)

    def value(self, observation):
        return self.value_head(self.observation_space.flatten_sample(observation))

    def action_and_value(self, observation, *, key):
        action_dist = self.action_dist(observation)
        action, log_prob = action_dist.sample_and_log_prob(key)

        return action, self.value(observation), log_prob

    def evaluate_action(self, observation, action):
        value = self.value(observation)

        action_dist = self.action_dist(observation)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return value, log_prob, entropy
```
