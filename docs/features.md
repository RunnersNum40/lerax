---
title: Features
description: Features of Lerax
---

## TensorBoard Logging

To log metrics and visualizations using TensorBoard simply pass a path or `True` to the `tb_log` argument of the `learn` method of any algorithm.

- If `tb_log` is a string, it is used as the log directory.
- If `tb_log` is `True`, a default log directory is created.

```py
from jax import random as jr

from lerax.algorithm import PPO
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
    tb_log=True, # Or tb_log="logs/name"
)
```

## Showing a Progress Bar

To show a progress bar during training, set `show_progress_bar` to `True` in the `learn` method of any algorithm.

```py
from jax import random as jr

from lerax.algorithm import PPO
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
    show_progress_bar=True,
)
```

Note that that progress bars will not work if the entire `learn` method is run in a JIT-compiled region.
This includes using `equinox.filter_jit` or `jax.jit` on the `learn` method or other Jax transformations such as `jax.vmap` or `jax.pmap`.

```py
import equinox as eqx
policy = eqx.filter_jit(algo.learn)(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
    show_progress_bar=True, # Will not show progress bar
)
```

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
