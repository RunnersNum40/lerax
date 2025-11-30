---
title: Features
description: Features of Lerax
---

## TensorBoard Logging

To log metrics to TensorBoard during training, you can use the `TensorBoardCallback` provided in `lerax.callback`.
Configure the callback with the desired log directory or pick a name automatically from the policy and environment, and pass it to the `learn` method of your algorithm.

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import TensorBoardCallback
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
    callback=TensorBoardCallback(env=env, policy=policy),
)
```

## Showing a Progress Bar

To show a progress bar during training, you can use the `ProgressBarCallback` provided in `lerax.callback`.
Create a `ProgressBarCallback` instance with the total number of timesteps and pass it to the `learn` method of the algorithm.

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ProgressBarCallback
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
    callback=ProgressBarCallback(2**16),
)
```

Note that that progress bars will not work if the callback is created a JIT-compiled region. It must be created outside and passed in. This includes Jax transformations such as `grad`, and `vmap`.

## Gymnasium Compatibility

Lerax provides compatibility wrappers for [Gymnasium environments](https://gymnasium.farama.org/) via `lerax.compatibility.gym`.
The `GymToLeraxEnv` wrapper converts Gymnasium environments to Lerax-compatible environments.
The `LeraxToGymEnv` wrapper converts Lerax environments to Gymnasium-compatible environments.

At the moment multi-environment support is not provided for Gymnasium environments.
You must specify `num_envs=1` for the algorithm you use.

```py
import gymnasium as gym
from jax import random as jr

from lerax.algorithm import PPO
from lerax.compatibility.gym import GymToLeraxEnv
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

gym_env = gym.make("CartPole-v1")
env = GymToLeraxEnv(gym_env)
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO(num_envs=1)  # Vectorization is not supported for Gym environments

policy = algo.learn(env, policy, total_timesteps=2**16, key=learn_key)
```

## Gymnax Compatibility

Lerax provides compatibility wrappers for Gymnax environments via `lerax.compatibility.gymnax`.
The `GymnaxToLeraxEnv` wrapper converts Gymnax environments to Lerax-compatible environments.
The `LeraxToGymnaxEnv` wrapper converts Lerax environments to Gymnax-compatible environments.

```py
import gymnax
from jax import random as jr

from lerax.algorithm import PPO
from lerax.compatibility.gymnax import GymnaxToLeraxEnv
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

gymnax_env, params = gymnax.make("CartPole-v1")
env = GymnaxToLeraxEnv(gymnax_env, params)
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()

policy = algo.learn(env, policy, total_timesteps=2**16, key=learn_key)
```

## Rendering Environments

Rendering of environments is supported via the `render_states` and `render_stacked` methods of Lerax environments.

- `render_states` takes a list of states and renders them as an animation.
- `render_stacked` takes a state with batched dimensions and renders them as an animation.

```py
from jax import random as jr

from lerax.env import CartPole

env = CartPole()

key, reset_key = jr.split(jr.key(0), 2)
states = [env.initial(key=reset_key)]
for _ in range(1024):
    key, action_key, transition_key = jr.split(key, 3)
    action = env.action_space.sample(action_key)
    states.append(env.transition(states[-1], action, key=transition_key))

env.render_states(states, dt=1 / 60) # (1)!
```

1. Passing `dt` specifies the time interval between frames in seconds.

## Saving and Loading Policies

Saving and loading policies is supported via the `serialize` and `deserialize` methods of Lerax policies.
The serialize method takes a static path, a format string and arguments, or a callable and arguments and saves the policy to a `.eqx` file at the specified location.

For example, to save this policy:

```py
from jax import random as jr

from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
```

### Static path

```py
policy.serialize("policy.eqx")
```

### Format string and arguments

```py
policy.serialize("policy_{}_epoch_{epoch}.eqx", 1, epoch=10)
```

### Function and arguments

```py
policy.serialize(
    lambda base_path, n, epoch: f"{base_path}_{n}_epoch_{epoch}.eqx",
    "policy",
    1,
    epoch=10,
)
```
