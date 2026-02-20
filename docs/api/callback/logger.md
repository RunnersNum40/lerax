---
title: Logging Callback
---

::: lerax.callback.LoggingCallbackStepState

::: lerax.callback.AbstractLoggingBackend
    options:
        members: ["open", "log_scalars", "log_video", "close"]

::: lerax.callback.TensorBoardBackend
    options:
        members: ["__init__", "log_scalars", "log_video", "close"]

::: lerax.callback.WandbBackend
    options:
        members: ["__init__", "log_scalars", "log_video", "close"]

::: lerax.callback.ConsoleBackend
    options:
        members: ["log_scalars", "log_hparams", "close"]

::: lerax.callback.LoggingCallback
    options:
        members: ["__init__", "reset", "step_reset", "on_step", "on_iteration", "on_training_start", "on_training_end"]
