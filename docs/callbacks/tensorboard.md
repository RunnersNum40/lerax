---
title: TensorBoard Callback
description: Log training metrics to TensorBoard.
---

# TensorBoardCallback

`TensorBoardCallback` logs training statistics to TensorBoard via `tensorboardX.SummaryWriter`.
It is JIT-safe through debug callbacks, but must be **constructed outside** any JIT-compiled function.

## Logged metrics

At the end of each iteration (`on_iteration`), it logs:

- `episode/return`
  Exponential moving average of episode returns over all environments.

- `episode/length`
  Exponential moving average of episode lengths.

- `train/*`
  All scalars in `training_log` (from the algorithm), plus:

- `train/learning_rate` (if present in the Optax optimizer state; otherwise `NaN`).

NaN or Inf scalar values are rejected with an error.

The smoothing factor for the exponential moving averages is controlled by `alpha` (default `0.9`).

## Initialization and log directory

```py
from lerax.callback import TensorBoardCallback

callback = TensorBoardCallback(
    name=None,   # optional; log directory name
    env=env,     # optional; used for default name
    policy=policy,  # optional; used for default name
    alpha=0.9,   # EMA smoothing for episode stats
)
```

If `name` is `None`, a directory under `logs/` is created using the current time and available names:

- `"logs/{policy.name}_{env.name}_{time}"`
- `"logs/{env.name}_{time}"` or `"logs/{policy.name}_{time}"`
- `"logs/training_{time}"` as a fallback.

Behind the scenes, the callback uses a `JITSummaryWriter` wrapper and `callback_with_numpy_wrapper` so that logging can be triggered from within JAX-transformed code while the actual I/O happens on the Python side.

## Video recording

`TensorBoardCallback` can periodically record evaluation videos and log them to
TensorBoard under `eval/video`. Set `video_interval` to a positive integer to
enable it:

```py
callback = TensorBoardCallback(
    env=env,
    policy=policy,
    video_interval=10,      # record every 10 iterations
    video_num_steps=256,    # 256 environment steps per video
    video_width=640,        # render width in pixels
    video_height=480,       # render height in pixels
    video_fps=50.0,         # playback fps
)
```

At each recording iteration the callback runs an eager eval rollout with the
current policy using `jax.debug.callback` and writes the video with
`SummaryWriter.add_video`. If rendering fails for any reason a warning is
emitted and training continues uninterrupted.

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
