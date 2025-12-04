---
title: Compatibility
description: Use Lerax together with Gymnasium, Gymnax, and Stable Baselines3.
---

# Compatibility

Lerax provides wrappers to interoperate with other RL libraries and environment collections.

Wrappers come in two directions:

- External to Lerax: use third-party environments with Lerax algorithms and policies.
- Lerax to External: use Lerax environments with third-party algorithms.

## Gymnasium

Lerax integrates with [Gymnasium](https://gymnasium.farama.org/) via the [`lerax.compatibility.gym`](api/compatibility/gymnasium.md) module.

### Using Gymnasium environments in Lerax

`GymToLeraxEnv` adapts a Gymnasium `gym.Env` to the functional Lerax environment API and exposes it as an `AbstractEnv`.
This is useful when you want to reuse existing Gymnasium environments with Lerax algorithms, accepting that you lose full JAX performance.
The `GymToLeraxEnv` uses [`jax.experimental.io_callback`](https://docs.jax.dev/en/latest/external-callbacks.html#exploring-io-callback) to call the Python `reset` and `step` methods, which means you do not get the full benefits of JAX such as `jax.jit` and `jax.vmap` over vectorized environments.

!!! warning
    Due to the use of `io_callback`, using `GymToLeraxEnv` is not compatible with `jax.vmap` or `jax.pmap` which may be used internally by some Lerax algorithms for vectorization.
    To avoid issues, when using `GymToLeraxEnv`, ensure that the Lerax algorithm is configured with `num_envs=1`.

Example:

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

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
)
```

### Using Lerax environments from Gymnasium / SB3

`LeraxToGymEnv` exposes a Lerax `AbstractEnv` as a Gymnasium-compatible `gym.Env`:

This is designed primarily for using Lerax environments with Gymnasium-style algorithms such as [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/):

```py
from stable_baselines3 import PPO

from lerax.compatibility.gym import LeraxToGymEnv
from lerax.env import CartPole

env = LeraxToGymEnv(CartPole())
model = PPO("MlpPolicy", env, tensorboard_log="logs")
model.learn(total_timesteps=2**16, progress_bar=True)
```

Since the SB3 training loop is pure Python, the Lerax environment is executed on the host, with JAX used only inside the environment implementation.
The performance will generally be lower than using Lerax algorithms directly, but this allows you to use existing Gymnasium-style algorithms with Lerax environments.

## Gymnax

Lerax integrates with [Gymnax](https://github.com/RobertTLange/gymnax) via the [`lerax.compatibility.gymnax`](api/compatibility/gymnax.md) module.

### Using Gymnax environments with Lerax

`GymnaxToLeraxEnv` adapts a Gymnax environment to the functional Lerax environment API and exposes it as an `AbstractEnv`.
Because Gymnax environments are implemented in JAX, wrapped environments retain full JAX compatibility and performance.

Example:

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

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
)
```
