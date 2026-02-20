---
title: Callbacks
description: Side-effect hooks for monitoring and controlling training.
---

# Callbacks

Callbacks let you monitor and control training without modifying algorithms.
They are Equinox modules called from the training loop and are designed to be JAX-friendly (I/O goes through JAX debug callbacks).

All callbacks subclass [`AbstractCallback`](../api/callback/index.md) and implement some or all of:

- `reset` / `step_reset` — called before training and before each rollout.
- `on_step` — called after each environment step.
- `on_iteration` — called after each training iteration / update.
- `on_training_start` / `on_training_end` — called at the start and end of training.
- `continue_training` — optional early-stopping hook.

Each hook receives a context object (`ResetContext`, `StepContext`, `IterationContext`, `TrainingContext`) containing the current environment, policy, optimizer state, training log, callback state, and a `locals` dict.

## Using callbacks with algorithms

Algorithms accept either:

- A single callback instance, or
- A Python list of callbacks (internally wrapped in a `CallbackList`)

via the `callback` argument to `learn`.

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import LoggingCallback, ProgressBarCallback, TensorBoardBackend
from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()

callbacks = [
    ProgressBarCallback(total_timesteps=2**16, env=env, policy=policy),
    LoggingCallback(TensorBoardBackend()),
]

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
    callback=callbacks,
)
```

## Built-in callbacks

- [`ProgressBarCallback`](progress_bar.md):
  Rich-based progress bar showing iterations, elapsed/remaining time, and iterations per second.

- [`LoggingCallback`](logging.md):
  Logs training metrics (learning rate, training log entries, episode return/length EMAs) to TensorBoard, Aim, or Weights & Biases via a pluggable backend.

- `CallbackList`:
  Aggregates multiple callbacks and forwards all hooks to each one. Used automatically when you pass a list of callbacks.

- `EmptyCallback`:
  No-op callback that can be used as a placeholder or default.
