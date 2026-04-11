---
title: Curriculum Learning
description: How to vary environment parameters during training using scheduled and adaptive curricula.
---

# Curriculum Learning

Curricula modify environment parameters over the course of training. In Lerax, curricula are callbacks that update fields on `state.env` between iterations.

## Scheduled Curriculum

A `ScheduledCurriculum` modifies an environment field on a fixed schedule based on iteration count.

```py
from jax import random as jr

from lerax.algorithm import PPO
from lerax.curriculum import ScheduledCurriculum, linear_schedule
from lerax.env.classic_control import Pendulum
from lerax.policy import MLPActorCriticPolicy

env = Pendulum()
policy = MLPActorCriticPolicy(env=env, key=jr.key(0))
algo = PPO()

curriculum = ScheduledCurriculum(
    where=lambda env: env.m,  # (1)!
    schedule_fn=linear_schedule(start=0.5, end=2.0, total=500),  # (2)!
)

policy = algo.learn(
    env, policy, total_timesteps=2**18, key=jr.key(1), callback=curriculum
)
```

1. `where` selects which field on the environment to modify. Any array-valued field works.
2. `linear_schedule` linearly interpolates from `start` to `end` over `total` iterations, clamped outside the range.

### Multiple Fields

Compose multiple schedules with `CallbackList`:

```py
from lerax.callback import CallbackList
from lerax.curriculum import ScheduledCurriculum, linear_schedule, step_schedule

curriculum = CallbackList(callbacks=[
    ScheduledCurriculum(
        where=lambda env: env.m,
        schedule_fn=linear_schedule(start=0.5, end=2.0, total=500),
    ),
    ScheduledCurriculum(
        where=lambda env: env.g,
        schedule_fn=step_schedule(
            values=[5.0, 7.0, 9.8],
            boundaries=[200, 400],
        ),
    ),
])
```

### Schedule Functions

| Function | Behavior |
|---|---|
| `linear_schedule(start, end, total)` | Linear interpolation, clamped outside `[0, total]` |
| `step_schedule(values, boundaries)` | Discrete jumps at iteration boundaries |
| `cosine_schedule(start, end, total)` | Cosine annealing from `start` to `end` |

All schedule functions return a JAX-compatible callable `iteration_count -> value`.

## Adaptive Curriculum

Adaptive curricula track a user-defined performance metric and modify the environment based on it. `AbstractAdaptiveCurriculum` handles the metric tracking; subclasses implement `apply_curriculum` to decide how the metric drives parameter changes.

### LevelCurriculum

`LevelCurriculum` is the built-in concrete implementation. It steps through a sequence of parameter values, advancing to the next when the metric exceeds a threshold.

```py
from jax import numpy as jnp

from lerax.curriculum import LevelCurriculum

curriculum = LevelCurriculum(
    where=lambda env: env.max_speed,
    levels=jnp.array([4.0, 6.0, 8.0]),  # (1)!
    metric_fn=lambda done, reward, locals: reward,  # (2)!
    threshold=100.0,  # (3)!
    smoothing=0.05,  # (4)!
)
```

1. Array of parameter values for each level. Training starts at index 0.
2. Called every step with `(done, reward, locals_dict)`. The return value is accumulated per episode and tracked as an exponential moving average.
3. When the running metric exceeds this value, the curriculum advances to the next level.
4. EMA smoothing factor. Higher values respond faster to recent performance.

### Custom Adaptive Curricula

Subclass `AbstractAdaptiveCurriculum` to implement custom adaptation logic. The base class handles metric tracking in `on_step` and EMA smoothing in `on_iteration`. You only need to implement `apply_curriculum`:

```py
from lerax.curriculum import AbstractAdaptiveCurriculum

class MyCurriculum(AbstractAdaptiveCurriculum):
    def apply_curriculum(self, state, callback_state):
        # callback_state.running_metric has the EMA of your metric
        # callback_state.level tracks the current level
        # Modify state.env however you like via eqx.tree_at
        return state, callback_state
```

### Custom Metrics

The `metric_fn` receives three arguments at every step:

- `done`: boolean, whether the episode just ended
- `reward`: scalar reward at this step
- `locals`: dictionary with full transition details (`observation`, `action`, `next_env_state`, etc.)

Examples:

```py
# Episode return (sum of rewards)
metric_fn = lambda done, reward, locals: reward

# Binary success (reward > 0 at episode end)
metric_fn = lambda done, reward, locals: jnp.where(done, (reward > 0).astype(float), 0.0)
```
