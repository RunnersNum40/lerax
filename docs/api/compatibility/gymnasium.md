---
title: Gymnasium
description: Wrapper classes and space conversion utilities for Gymnasium.
---

::: lerax.compatibility.gym.GymToLeraxEnv
    options:
        members: ["name", "action_space", "observation_space", "__init__", "initial", "transition", "observation", "reward", "terminal", "truncate", "state_info", "transition_info", "default_renderer", "render", "render_stacked", "reset", "step", "close"]

::: lerax.compatibility.gym.LeraxToGymEnv
    options:
        members: true

::: lerax.compatibility.gym.gym_space_to_lerax_space
    options:
        annotations_path: "full"

::: lerax.compatibility.gym.lerax_to_gym_space
    options:
        annotations_path: "full"
