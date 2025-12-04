from __future__ import annotations

from datetime import datetime

import optax
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key, Scalar, ScalarLike
from tensorboardX import SummaryWriter

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy
from lerax.utils import callback_with_numpy_wrapper

from .base_callback import (
    AbstractCallback,
    AbstractCallbackStepState,
    EmptyCallbackState,
    IterationContext,
    ResetContext,
    StepContext,
    TrainingContext,
)


class JITSummaryWriter:
    """
    A wrapper around `tensorboardX.SummaryWriter` with a JIT compatible interface.

    Attributes:
        summary_writer: The underlying SummaryWriter instance.

    Args:
        log_dir: Directory to save TensorBoard logs. If None, uses default.
    """

    summary_writer: SummaryWriter

    def __init__(self, log_dir: str | None = None):
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(
        self,
        tag: str,
        scalar_value: ScalarLike,
        global_step: Int[ArrayLike, ""] | None = None,
        walltime: Float[ArrayLike, ""] | None = None,
    ):
        """
        Add a scalar value to the summary writer.
        """
        callback_with_numpy_wrapper(self.summary_writer.add_scalar)(
            tag, scalar_value, global_step, walltime
        )

    def add_dict(
        self,
        prefix: str,
        scalars: dict[str, Scalar],
        *,
        global_step: Int[ArrayLike, ""] | None = None,
        walltime: Float[ArrayLike, ""] | None = None,
    ) -> None:
        """
        Log a dictionary of **scalar** values.
        """

        if prefix:
            scalars = {f"{prefix}/{k}": v for k, v in scalars.items()}

        for tag, value in scalars.items():
            self.add_scalar(tag, value, global_step=global_step, walltime=walltime)


class TensorBoardCallbackStepState(AbstractCallbackStepState):
    """
    State for TensorBoardCallback.

    Records cumulative episode returns and lengths, and the exponential moving
    average of them over episodes.

    Attributes:
        step: Current training step.
        episode_return: Cumulative return for the current episode.
        episode_length: Length of the current episode.
        episode_done: Boolean indicating if the current episode is done.
        average_return: Exponential moving average of episode returns.
        average_length: Exponential moving average of episode lengths.

    Args:
        step: Current training step.
        episode_return: Cumulative return for the current episode.
        episode_length: Length of the current episode.
        episode_done: Boolean indicating if the current episode is done.
        average_return: Exponential moving average of episode returns.
        average_length: Exponential moving average of episode lengths.
    """

    step: Int[Array, ""]

    episode_return: Float[Array, ""]
    episode_length: Int[Array, ""]
    episode_done: Bool[Array, ""]

    average_return: Float[Array, ""]
    average_length: Float[Array, ""]

    def __init__(
        self,
        step: Int[Array, ""],
        episode_return: Float[ArrayLike, ""],
        episode_length: Int[ArrayLike, ""],
        episode_done: Bool[ArrayLike, ""],
        average_return: Float[ArrayLike, ""],
        average_length: Float[ArrayLike, ""],
    ):
        self.step = jnp.asarray(step)
        self.episode_return = jnp.asarray(episode_return)
        self.episode_length = jnp.asarray(episode_length)
        self.episode_done = jnp.asarray(episode_done)
        self.average_return = jnp.asarray(average_return)
        self.average_length = jnp.asarray(average_length)

    @classmethod
    def initial(cls) -> TensorBoardCallbackStepState:
        return cls(
            jnp.array(0, dtype=int),
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
            jnp.array(False, dtype=bool),
            jnp.array(0.0, dtype=float),
            jnp.array(0.0, dtype=float),
        )

    def next(
        self, reward: Float[Array, ""], done: Bool[Array, ""], alpha: float
    ) -> TensorBoardCallbackStepState:
        average_return = lax.select(
            done,
            alpha * self.episode_return + (1.0 - alpha) * self.average_return,
            self.average_return,
        )
        average_length = lax.select(
            done,
            alpha * self.episode_length.astype(float)
            + (1.0 - alpha) * self.average_length,
            self.average_length,
        )

        return TensorBoardCallbackStepState(
            self.step + 1,
            self.episode_return * (1.0 - self.episode_done.astype(float)) + reward,
            self.episode_length * (1 - self.episode_done.astype(int)) + 1,
            done,
            average_return,
            average_length,
        )


class TensorBoardCallback(
    AbstractCallback[EmptyCallbackState, TensorBoardCallbackStepState]
):
    """
    Callback for recording training statistics to TensorBoard.

    Each training iteration, the following statistics are logged:
        - episode/return: The exponential moving average of episode returns.
        - episode/length: The exponential moving average of episode lengths.
        - train/:
            - learning_rate: The current learning rate.
            - ...: Any other statistics in the training log.

    Note:
        If the callback is instantiated inside a JIT-compiled function, it may
        not work correctly.

    Attributes:
        tb_writer: The TensorBoard summary writer.
        alpha: Smoothing factor for exponential moving averages.

    Args:
        name: Name for the TensorBoard log directory. If None, a name
            is generated based on the current time, environment name, and policy name.
        env: The environment being trained on. Used for naming if `name` is None.
        policy: The policy being trained. Used for naming if `name` is None.
        alpha: Smoothing factor for exponential moving averages.
    """

    tb_writer: JITSummaryWriter
    alpha: float

    def __init__(
        self,
        name: str | None = None,
        env: AbstractEnvLike | None = None,
        policy: AbstractPolicy | None = None,
        alpha: float = 0.9,
    ):
        log_dir = "logs/"
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name is None:
            if env is not None:
                if policy is not None:
                    name = f"{log_dir}{policy.name}_{env.name}_{time}"
                else:
                    name = f"{log_dir}{env.name}_{time}"
            else:
                if policy is not None:
                    name = f"{log_dir}{policy.name}_{time}"
                else:
                    name = f"{log_dir}training_{time}"

        self.tb_writer = JITSummaryWriter(log_dir=name)
        self.alpha = alpha

    def reset(self, ctx: ResetContext, *, key: Key) -> EmptyCallbackState:
        return EmptyCallbackState()

    def step_reset(
        self, ctx: ResetContext, *, key: Key
    ) -> TensorBoardCallbackStepState:
        return TensorBoardCallbackStepState.initial()

    def on_step(self, ctx: StepContext, *, key: Key) -> TensorBoardCallbackStepState:
        return ctx.state.next(ctx.reward, ctx.done, self.alpha)

    def on_iteration(self, ctx: IterationContext, *, key: Key) -> EmptyCallbackState:
        log = ctx.training_log
        opt_state = ctx.opt_state

        log["learning_rate"] = optax.tree_utils.tree_get(
            opt_state, "learning_rate", jnp.nan
        )

        step_state = ctx.step_state

        last_step = step_state.step.sum()

        self.tb_writer.add_dict("train", log, global_step=last_step)
        self.tb_writer.add_scalar(
            "episode/return", step_state.average_return.mean(), last_step
        )
        self.tb_writer.add_scalar(
            "episode/length", step_state.average_length.mean(), last_step
        )

        return ctx.state

    def on_training_start(
        self, ctx: TrainingContext, *, key: Key
    ) -> EmptyCallbackState:
        return ctx.state

    def on_training_end(self, ctx: TrainingContext, *, key: Key) -> EmptyCallbackState:
        return ctx.state

    def continue_training(self, ctx: IterationContext, *, key: Key) -> Bool[Array, ""]:
        return jnp.array(True)
