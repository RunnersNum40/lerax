---
title: Making a Custom Policy
description: Example of creating a custom policy in Lerax reinforcement learning library.
status: draft
---

While Lerax provides several built-in policies, you may want to create your own custom policy to better suit your specific task.

### Policies are Equinox Modules

Lerax policies are [Equinox Modules](https://docs.kidger.site/equinox/api/module/module/) which means they are frozen [dataclasses](https://docs.python.org/3/library/dataclasses.html) that are automatically registered as [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html) so they can be used across JAX transformations.
All the fields and abstract methods of the base classes must be implemented in your custom policy or else you will get an error when trying to instantiate it.
They must be initialized in the `__init__` method and cannot be modified afterwards.

### Stateful vs Stateless Policies

Lerax policies can be either stateful or stateless.
Algorithms work on stateful policies only and will automatically convert stateless policies to stateful ones when needed.
For this reason stateless policies must implement `into_stateful`.
Typically this is done with a wrapper class that adds a dummy state.
For the built-in policy base classes this is already done for you.

All policy states must subclass `lerax.policy.AbstractPolicyState`.
This makes them an Equinox Module as well.

## A Custom Stateless Actor-Critic Policy

Let's create a simple custom stateless actor-critic policy.
API documentation is coming but for now reference the base class [`lerax.policy.AbstractStatelessActorCriticPolicy`](https://github.com/RunnersNum40/lerax/blob/main/lerax/policy/actor_critic/base_actor_critic.py#L19)

The fields that need to be specified and initialized are:

- `#!python action_space: AbstractSpace`: The action space of the environment.
- `#!python observation_space: AbstractSpace`: The observation space of the environment.
- `#!python name: str`: The name of the policy.

The methods that need to be implemented are:

- `#!python action_and_value(observation: ObsType, *, key: Key) -> tuple[ActType, Float, Float]`: Returns the action, value, and log probability of the action for a given observation.
- `#!python evaluate_action(observation: ObsType, action: ActType) -> tuple[Float, Float, Float]`: Returns the value, log probability, and entropy for a given observation and action.
- `#!python value(observation: ObsType) -> Float`: Returns the value for a given observation.

```py
import equinox as eqx
from jax import random as jr

from lerax.distribution import Categorical
from lerax.env import AbstractEnvLike
from lerax.policy import AbstractStatelessActorCriticPolicy
from lerax.space import AbstractSpace, Discrete


class ActorCriticPolicy(AbstractStatelessActorCriticPolicy):
    action_space: Discrete
    observation_space: AbstractSpace

    actor: eqx.nn.MLP
    critic: eqx.nn.MLP

    name: str = "CustomActorCriticPolicy"

    def __init__(self, env: AbstractEnvLike, key):
        assert isinstance(env.action_space, Discrete)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        actor_key, critic_key = jr.split(key)

        self.actor = eqx.nn.MLP(
            in_size=int(self.observation_space.flat_size), # (1)!
            out_size=int(self.action_space.n),
            width_size=64,
            depth=2,
            key=actor_key,
        )

        self.critic = eqx.nn.MLP(
            in_size=int(self.observation_space.flat_size),
            out_size="scalar",
            width_size=64,
            depth=2,
            key=critic_key,
        )

    def features(self, observation):
        """Helper function to convert observations into 1D vectors."""
        return self.observation_space.flatten_sample(observation) # (2)!

    def action_distribution(self, observation):
        """Helper function to get the action distribution for a given observation."""
        logits = self.actor(self.features(observation))
        return Categorical(logits=logits) # (3)!

    def action_and_value(self, observation, *, key):
        """Return action, value, and log probability of the action for a given observation."""
        action_dist = self.action_distribution(observation)
        value = self.value(self.features(observation))

        action = action_dist.sample(key=key)
        log_prob = action_dist.log_prob(action)

        return action, value, log_prob

    def evaluate_action(self, observation, action):
        """Return value, log probability, and entropy for a given observation and action."""
        action_dist = self.action_distribution(observation)
        value = self.value(self.features(observation))

        log_prob = action_dist.log_prob(action)

        return value, log_prob, action_dist.entropy()

    def value(self, observation):
        """Return the value for a given observation."""
        return self.critic(self.features(observation))
```

1. Lerax spaces provide `flatten_sample` to convert samples into 1D vectors.
  The `flat_size` property gives the size of the flattened vector.
2. Here we flatten the observation using the observation space's `flatten_sample` method.
3. The actor network outputs logits for a categorical distribution over discrete actions.
  Lerax distributions can be used to handle sampling, log probabilities, and entropy calculations.


Now we can train it on the CartPole environment:

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.env import CartPole

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = ActorCriticPolicy(env=env, key=policy_key)
algo = PPO()

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
    show_progress_bar=True,
    tb_log=True,
)
```
