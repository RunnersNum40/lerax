---
title: Env
description: Functional environment interfaces and state types.
---

# Env API

::: lerax.env.AbstractEnv
    options:
        members: ["name", "action_space", "observation_space", "__init__", "initial", "action_mask", "transition", "observation", "reward", "terminal", "truncate", "state_info", "transition_info", "default_renderer", "render", "render_stacked", "reset", "step"]

::: lerax.env.AbstractEnvState
    options:
        filters: ["!unwrapped"]
