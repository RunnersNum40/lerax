---
title: Recording to Video
description: Saving rendered rollouts as video files.
---

# Recording to Video

## 2D environments

`VideoRenderer` wraps a 2D renderer and records frames to a video file.
All drawing goes through the inner renderer; frames are captured on each `draw()` and written when `close()` is called.

Given a sequence of states (for example from `jax.lax.scan`):

```py
from lerax.render import VideoRenderer

renderer = VideoRenderer(
    env.default_renderer(),
    "cartpole.mp4",
    fps=60.0,
)

env.render_stacked(env_states, renderer=renderer, dt=1 / 60)
```

## MuJoCo environments

`MujocoVideoRenderer` renders MuJoCo scenes headlessly and records frames
to a video file. It works with any MuJoCo-based environment (standard MuJoCo
envs and Unitree G1 envs). It may take a concernable amount of time to JIT the rendering code, once it's JITted, rendering is very fast (much faster than real-time).

```py
from lerax.render.mujoco_renderer import MujocoVideoRenderer

renderer = MujocoVideoRenderer(
    env.mujoco_model,
    "humanoid.mp4",
    fps=50.0,
    width=800,
    height=600,
)

env.render_stacked(env_states, renderer=renderer)
```

For headless rendering without video recording (e.g. to capture individual
frames as arrays), use `MujocoRenderer` with `headless=True`:

```py
from lerax.render.mujoco_renderer import MujocoRenderer

renderer = MujocoRenderer(env.mujoco_model, headless=True)
renderer.open()
renderer.render(state.sim_state)
renderer.draw()
frame = renderer.as_array()  # numpy array of shape (H, W, 3)
renderer.close()
```

!!! note "Headless backend"
    Headless MuJoCo rendering requires an OpenGL backend that supports
    offscreen rendering. On Linux without a display, set `MUJOCO_GL=egl`
    (requires EGL support) or `MUJOCO_GL=osmesa` (software rendering).
