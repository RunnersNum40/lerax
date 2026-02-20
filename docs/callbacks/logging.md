---
title: Logging Callback
description: Log training metrics to TensorBoard, Weights & Biases, or the console.
---

# LoggingCallback

`LoggingCallback` logs training statistics to any supported backend via a pluggable `AbstractLoggingBackend`.
It is JIT-safe through debug callbacks, but must be **constructed outside** any JIT-compiled function.

## Run naming

By default a run name is generated from the algorithm, policy, and environment:

    {Algorithm}_{PolicyName}_{EnvName}_{timestamp}

For example: `PPO_MLPActorCriticPolicy_CartPole-v1_20260220_120000`.

Pass `name=` to `LoggingCallback` to override:

```py
LoggingCallback(backend, name="my-experiment")
```

## Logged metrics

At the end of each iteration (`on_iteration`), it logs:

- `episode/return`
  Exponential moving average of episode returns over all environments.

- `episode/length`
  Exponential moving average of episode lengths.

- `train/*`
  All scalars in `training_log` (from the algorithm), plus:

- `train/learning_rate` (if present in the Optax optimizer state; otherwise `NaN`).

The smoothing factor for the exponential moving averages is controlled by `alpha` (default `0.9`).

## Usage

```py
from lerax.callback import LoggingCallback, TensorBoardBackend

callback = LoggingCallback(
    backend=TensorBoardBackend(),
    alpha=0.9,
)
```

## Backends

All backends defer initialisation to their `open` method, which `LoggingCallback`
calls automatically at training start with the run name.

### TensorBoardBackend

Writes to TensorBoard via `tensorboardX`.

```py
from lerax.callback import TensorBoardBackend

backend = TensorBoardBackend(
    log_dir="logs", # base directory for event files
)
```

### WandbBackend

Writes to [Weights & Biases](https://wandb.ai/).

```py
from lerax.callback import WandbBackend

backend = WandbBackend(
    project="my-project",
    config={"lr": 3e-4},  # optional hyperparameter dict
)
```

### ConsoleBackend

Prints metrics to the terminal using Rich. Useful for quick debugging without a logging server.

```py
from lerax.callback import ConsoleBackend

backend = ConsoleBackend()
```

## Video recording

`LoggingCallback` can periodically record evaluation videos and log them under `eval/video`. Set `video_interval` to a positive integer to enable it:

```py
callback = LoggingCallback(
    backend=TensorBoardBackend(),
    video_interval=10,      # record every 10 iterations
    video_num_steps=256,    # 256 environment steps per video
    video_width=640,        # render width in pixels
    video_height=480,       # render height in pixels
    video_fps=50.0,         # playback fps
)
```

At each recording iteration the callback runs an eager eval rollout with the
current policy using `jax.debug.callback` and forwards the frames to the
backend. If rendering fails for any reason a warning is emitted and training
continues uninterrupted.

Two renderer backends are supported:

- **MuJoCo** environments (whose `unwrapped` exposes a `mujoco_model` attribute)
  use `HeadlessMujocoRenderer` for off-screen rendering.
- **PyGame** environments (e.g. CartPole, MountainCar) use
  `HeadlessPygameRenderer`, which renders into an off-screen `pygame.Surface`
  without requiring a display server. The correct world transform is extracted
  from the environment's `default_renderer()`.

!!! note "Display initialisation"
    `HeadlessPygameRenderer` sets `SDL_VIDEODRIVER=dummy` before initialising
    pygame when no display is available. If pygame is already initialised with a
    real driver, the environment's `default_renderer()` may briefly open a
    window; this is harmless during training, where interactive pygame rendering
    is unlikely.
