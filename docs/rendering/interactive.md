---
title: Interactive Rendering
description: Rendering Lerax environments in a window.
---

# Interactive Rendering

## Single-episode rollout

A simple way to visualize an environment is to use its `default_renderer()` and call `render` in a Python loop.

```py
from jax import random as jr

from lerax.env import CartPole

env = CartPole()

key, reset_key = jr.split(jr.key(0), 2)
state = env.initial(key=reset_key)
renderer = env.default_renderer()

renderer.open()

for _ in range(256):
    key, action_key, transition_key = jr.split(key, 3)
    action = env.action_space.sample(action_key)
    state = env.transition(state, action, key=transition_key)
    env.render(state, renderer)

renderer.close()
```

Notes:

- `default_renderer()` typically returns a `PygameRenderer` configured for that environment’s coordinate system.
- `render` clears and redraws the current frame, then calls `renderer.draw()` internally.

## Rendering a scanned rollout

`render_stacked` is convenient when the rollout comes from `jax.lax.scan` (e.g. a pure JAX rollout used for training or evaluation).

```py
from jax import lax
from jax import random as jr

from lerax.env import CartPole

env = CartPole()

def step(env_state, key):
    action_key, transition_key, terminal_key, reset_key = jr.split(key, 4)

    action = env.action_space.sample(action_key)
    env_state = env.transition(env_state, action, key=transition_key)
    done = env.terminal(env_state, key=terminal_key) | env.truncate(env_state)

    env_state = lax.cond(
        done,
        lambda: env.initial(key=reset_key),
        lambda: env_state,
    )

    return env_state, env_state

reset_key, rollout_key = jr.split(jr.key(0), 2)
state0 = env.initial(key=reset_key)

_, env_states = lax.scan(step, state0, jr.split(rollout_key, 1024))


env.render_stacked(env_states, dt=1 / 60)
```

Internally, `render_stacked` uses `lerax.utils.unstack_pytree` and forwards each unstacked state to `render_states`.
