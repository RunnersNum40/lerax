---
title: Callback
---

::: lerax.callback.AbstractCallbackStepState
::: lerax.callback.AbstractCallbackState

::: lerax.callback.ResetContext
::: lerax.callback.StepContext
::: lerax.callback.IterationContext
::: lerax.callback.TrainingContext

::: lerax.callback.AbstractCallback
    options:
        members: ["__init__", "reset", "step_reset", "on_step", "on_iteration", "on_training_start", "on_training_end"]

::: lerax.callback.AbstractStatelessCallback
    options:
        members: ["__init__", "on_step", "on_iteration", "on_training_start", "on_training_end"]

::: lerax.callback.AbstractStepCallback
    options:
        members: ["__init__", "step_reset", "on_step"]

::: lerax.callback.AbstractIterationCallback
    options:
        members: ["__init__", "reset", "on_iteration"]

::: lerax.callback.AbstractTrainingCallback
    options:
        members: ["__init__", "reset", "on_training_start", "on_training_end"]
