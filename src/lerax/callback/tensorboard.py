from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import equinox as eqx
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key, Scalar, ScalarLike
from tensorboardX import SummaryWriter

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy
from lerax.utils import callback_with_numpy_wrapper, callback_wrapper

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

    def __init__(self, log_dir: str | Path | None = None):
        if log_dir is None:
            self.summary_writer = SummaryWriter()
        else:
            self.summary_writer = SummaryWriter(log_dir=Path(log_dir).as_posix())

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
        episode_return = (
            self.episode_return * (1.0 - self.episode_done.astype(float)) + reward
        )
        episode_length = self.episode_length * (1 - self.episode_done.astype(int)) + 1

        average_return = lax.select(
            done,
            alpha * episode_return + (1.0 - alpha) * self.average_return,
            self.average_return,
        )
        average_length = lax.select(
            done,
            alpha * episode_length.astype(float) + (1.0 - alpha) * self.average_length,
            self.average_length,
        )

        return TensorBoardCallbackStepState(
            self.step + 1,
            episode_return,
            episode_length,
            done,
            average_return,
            average_length,
        )


def _make_video_recorder(
    video_interval: int,
    video_num_steps: int,
    video_width: int,
    video_height: int,
    video_fps: float,
    writer: SummaryWriter,
) -> Callable[..., None]:
    """Create a JIT-safe video recording callback function.

    Returns a ``callback_wrapper``-decorated function that runs an eager eval
    rollout and logs the resulting video to TensorBoard.

    Both MuJoCo-based environments and PyGame-based environments (e.g. CartPole,
    MountainCar) are supported. MuJoCo environments are detected by the presence
    of a ``mujoco_model`` attribute on the unwrapped environment. For PyGame
    environments, ``default_renderer()`` is called to extract the correct world
    transform before constructing a ``HeadlessPygameRenderer``.

    Args:
        video_interval: Record a video every this many iterations.
        video_num_steps: Number of environment steps per video.
        video_width: Render width in pixels.
        video_height: Render height in pixels.
        video_fps: Playback frames per second.
        writer: The underlying ``SummaryWriter`` to log videos to.
    """

    @partial(callback_wrapper, ordered=True)
    def record_video(
        env: AbstractEnvLike,
        policy: AbstractPolicy,
        step: Int[Array, ""],
        iteration_count: Int[Array, ""],
        key: Key[Array, ""],
    ) -> None:
        import numpy as np

        if int(iteration_count) % video_interval != 0:
            return

        try:
            unwrapped = env.unwrapped
            mujoco_model = getattr(unwrapped, "mujoco_model", None)

            if mujoco_model is not None:
                from lerax.render.mujoco_renderer import HeadlessMujocoRenderer

                mujoco_renderer = HeadlessMujocoRenderer(
                    mujoco_model,
                    width=video_width,
                    height=video_height,
                )
                renderer = mujoco_renderer

                def render_frame(env_state) -> np.ndarray:
                    mujoco_renderer.render(env_state.data)
                    rgb, _ = mujoco_renderer.read_pixels()
                    return rgb

            else:
                import os

                import pygame as pg

                from lerax.render.pygame_renderer import (
                    HeadlessPygameRenderer,
                    PygameRenderer,
                )

                already_init = pg.get_init()
                old_driver = os.environ.get("SDL_VIDEODRIVER")
                if not already_init:
                    os.environ["SDL_VIDEODRIVER"] = "dummy"

                try:
                    template = unwrapped.default_renderer()
                except Exception:
                    return

                if not isinstance(template, PygameRenderer):
                    template.close()
                    return

                pygame_renderer = HeadlessPygameRenderer(
                    width=template.width,
                    height=template.height,
                    background_color=template.background_color,
                    transform=template.transform,
                )
                renderer = pygame_renderer
                template.close()

                if not already_init:
                    if old_driver is None:
                        os.environ.pop("SDL_VIDEODRIVER", None)
                    else:
                        os.environ["SDL_VIDEODRIVER"] = old_driver

                def render_frame(env_state) -> np.ndarray:
                    env.render(env_state, pygame_renderer)
                    return np.asarray(pygame_renderer.as_array())

            init_key, policy_key, rollout_key = jr.split(key, 3)
            env_state = env.initial(key=init_key)
            policy_state = policy.reset(key=policy_key)

            frames: list[np.ndarray] = []
            for _ in range(video_num_steps):
                obs_key, mask_key, step_key, reset_key, rollout_key = jr.split(
                    rollout_key, 5
                )
                observation = env.observation(env_state, key=obs_key)
                action_mask = env.action_mask(env_state, key=mask_key)
                policy_state, action = policy(
                    policy_state, observation, key=step_key, action_mask=action_mask
                )
                env_state = env.transition(env_state, action, key=step_key)

                frames.append(render_frame(env_state))

                done = env.terminal(env_state, key=step_key) | env.truncate(env_state)
                if bool(done):
                    env_state = env.initial(key=reset_key)
                    policy_state = policy.reset(key=reset_key)

            renderer.close()

            video = np.stack(frames)
            video = np.transpose(video, (0, 3, 1, 2))
            video = video[np.newaxis, ...]
            writer.add_video("eval/video", video, int(step), fps=video_fps)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Video recording failed at iteration {int(iteration_count)}: {exc}",
                stacklevel=2,
            )

    return record_video


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

    When ``video_interval`` is set to a positive integer, an evaluation video is
    recorded every ``video_interval`` iterations and logged under ``eval/video``
    in TensorBoard. Both MuJoCo-based environments and PyGame-based environments
    (e.g. CartPole, MountainCar) are supported.

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
        log_dir: Base directory for TensorBoard logs.
        video_interval: Record an evaluation video every this many iterations.
            Set to 0 (default) to disable video recording.
        video_num_steps: Number of environment steps per recorded video.
        video_width: Width of the rendered video frames in pixels.
        video_height: Height of the rendered video frames in pixels.
        video_fps: Playback frames per second for the recorded video.
    """

    tb_writer: JITSummaryWriter
    alpha: float
    _record_video_fn: Callable[..., None] | None = eqx.field(static=True)

    def __init__(
        self,
        name: str | None = None,
        env: AbstractEnvLike | None = None,
        policy: AbstractPolicy | None = None,
        alpha: float = 0.9,
        log_dir: str | Path = "logs",
        video_interval: int = 0,
        video_num_steps: int = 128,
        video_width: int = 640,
        video_height: int = 480,
        video_fps: float = 50.0,
    ):
        log_dir = Path(log_dir)
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name is None:
            if env is not None:
                if policy is not None:
                    name = f"{policy.name}_{env.name}_{time}"
                else:
                    name = f"{env.name}_{time}"
            else:
                if policy is not None:
                    name = f"{policy.name}_{time}"
                else:
                    name = f"training_{time}"

        path = log_dir / name

        self.tb_writer = JITSummaryWriter(log_dir=path)
        self.alpha = alpha

        if video_interval > 0:
            self._record_video_fn = _make_video_recorder(
                video_interval,
                video_num_steps,
                video_width,
                video_height,
                video_fps,
                self.tb_writer.summary_writer,
            )
        else:
            self._record_video_fn = None

    def reset(self, ctx: ResetContext, *, key: Key[Array, ""]) -> EmptyCallbackState:
        return EmptyCallbackState()

    def step_reset(
        self, ctx: ResetContext, *, key: Key[Array, ""]
    ) -> TensorBoardCallbackStepState:
        return TensorBoardCallbackStepState.initial()

    def on_step(
        self, ctx: StepContext, *, key: Key[Array, ""]
    ) -> TensorBoardCallbackStepState:
        return ctx.state.next(ctx.reward, ctx.done, self.alpha)

    def on_iteration(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        log = ctx.training_log
        opt_state = ctx.opt_state

        log["learning_rate"] = optax.tree_utils.tree_get(
            opt_state,
            "learning_rate",
            jnp.nan,
            filtering=lambda _, value: isinstance(value, jnp.ndarray),
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

        if self._record_video_fn is not None:
            video_key, key = jr.split(key)
            self._record_video_fn(
                ctx.env, ctx.policy, last_step, ctx.iteration_count, video_key
            )

        return ctx.state

    def on_training_start(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        return ctx.state

    def on_training_end(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        return ctx.state

    def continue_training(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return jnp.array(True)
