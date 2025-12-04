---
title: Rendering
description: Rendering and recording Lerax environments.
---

# Rendering

Lerax environments expose a small, Python-side rendering API for visualization and video recording.

!!! warning "JIT Safety"
    Rendering is not JIT-safe. Call rendering code outside `jax.jit`, `jax.vmap`, etc.

To install rendering dependencies:

```bash
pip install "lerax[render]"
```

## Environment-level API

All environments (and wrappers) implement the `AbstractEnvLike` interface and provide:

- `default_renderer()` — construct a suitable renderer (usually a `PygameRenderer`).
- `render(state, renderer)` — draw a single frame.
- `render_states(states, renderer="auto", dt=0.0)` — render a Python sequence of states as an animation.
- `render_stacked(states, renderer="auto", dt=0.0)` — render a batched/stacked state PyTree (e.g. from `jax.lax.scan`).

If `renderer="auto"`, the environment’s `default_renderer()` is used.

`dt` is a Python-side sleep (in seconds) between frames; it does not change the environment dynamics.

## Guide topics

- [Interactive rendering](interactive.md)
  How to open a window and render rollouts from Python or scanned traces.

- [Recording to video](video.md)
  Using `VideoRenderer` to capture frames from another renderer and write them to disk.

For API details see:

- [`lerax.render.AbstractRenderer` and concrete renderers](api/render/index.md)
- [`Color`](api/render/color.md) and [`Transform`](api/render/transform.md)
