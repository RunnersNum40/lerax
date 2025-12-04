---
title: Progress Bar Callback
description: Display a live Rich progress bar during training.
---

# ProgressBarCallback

`ProgressBarCallback` displays a terminal progress bar using [Rich](https://rich.readthedocs.io/).
It shows:

- Completed vs total iterations or timesteps.
- Elapsed and estimated remaining time.
- Iterations per second (human-readable units).

The callback is JIT-safe via debug callbacks, but it must be **constructed outside** any JIT-compiled function.

## Initialization

```py
from lerax.callback import ProgressBarCallback

callback = ProgressBarCallback(
    total_timesteps=2**16,
    name=None, # optional; bar title
    env=env, # optional; used for the bar title
    policy=policy, # optional; used for the bar title
)
```

If `name` is not provided, a default description is generated:

- `"Training {policy.name} on {env.name}"` when both are given.
- `"Training on {env.name}"` or `"Training {policy.name}"` when only one is given.
- `"Training"` otherwise.

## Behavior

- `reset` starts the underlying `Rich` progress bar.
- `on_step` counts steps in the current iteration.
- `on_iteration` updates the bar with the total steps taken in that iteration.
- `continue_training` always returns `True` (does not perform early stopping).

!!! warning
    `on_training_end` currently does not stop the bar explicitly due to ordering issues; the bar is typically cleared automatically when the process exits or the `Progress` object is collected.
    To manually stop the bar, you can call `callback._progress.stop()`.
