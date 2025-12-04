---
title: Recording to Video
description: Saving rendered rollouts as video files.
---

# Recording to Video

`VideoRenderer` wraps another renderer and records frames to a video file.
All drawing goes through the inner renderer; frames are captured on each `draw()` and written when `close()` is called.

## Basic usage

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
