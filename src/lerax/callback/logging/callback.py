from __future__ import annotations

import concurrent.futures
import dataclasses
import os
from collections.abc import Callable
from datetime import datetime
from functools import partial
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.env import AbstractEnvLike
from lerax.policy import AbstractPolicy
from lerax.utils import (
    callback_with_numpy_wrapper,
    callback_wrapper,
    filter_cond,
    filter_scan,
)

from ..base_callback import (
    AbstractCallback,
    AbstractCallbackStepState,
    EmptyCallbackState,
    IterationContext,
    ResetContext,
    StepContext,
    TrainingContext,
)
from .backend import AbstractLoggingBackend


def _extract_hparams(module: eqx.Module, prefix: str = "") -> dict[str, Any]:
    """
    Recursively extract scalar Python values from an Equinox module's fields.

    Traverses the module tree, collecting ``int``, ``float``, ``bool``, ``str``
    values and sequences of numbers. Private fields (leading ``_``) and JAX
    arrays are skipped.

    Args:
        module: Equinox module to inspect.
        prefix: Dot-separated key prefix for nested modules.

    Returns:
        Flat dict mapping dotted field paths to scalar values.
    """
    result: dict[str, Any] = {}
    for field in dataclasses.fields(module):
        if field.name.startswith("_"):
            continue
        value = getattr(module, field.name)
        key = f"{prefix}.{field.name}" if prefix else field.name
        if isinstance(value, bool):
            result[key] = value
        elif isinstance(value, (int, float)) and not eqx.is_array(value):
            result[key] = value
        elif isinstance(value, str):
            result[key] = value
        elif isinstance(value, eqx.Module):
            result.update(_extract_hparams(value, key))
        elif (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(v, (int, float)) and not eqx.is_array(v) for v in value)
        ):
            result[key] = str(value)
    return result


class LoggingCallbackStepState(AbstractCallbackStepState):
    """
    Per-environment step state for `LoggingCallback`.

    Tracks cumulative episode returns and lengths, along with exponential
    moving averages of those quantities updated at episode boundaries.

    Attributes:
        step: Total number of steps taken.
        episode_return: Cumulative return for the current (in-progress) episode.
        episode_length: Number of steps in the current episode.
        episode_done: Whether the previous step ended an episode.
        average_return: EMA of episode returns.
        average_length: EMA of episode lengths.

    Args:
        step: Total number of steps taken.
        episode_return: Cumulative return for the current episode.
        episode_length: Number of steps in the current episode.
        episode_done: Whether the previous step ended an episode.
        average_return: EMA of episode returns.
        average_length: EMA of episode lengths.
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
    def initial(cls) -> LoggingCallbackStepState:
        """Return a zeroed initial state."""
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
    ) -> LoggingCallbackStepState:
        """
        Advance the state by one environment step.

        Args:
            reward: Reward received at this step.
            done: Whether the episode ended at this step.
            alpha: EMA smoothing factor (weight on the new episode value).

        Returns:
            Updated step state.
        """
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

        return LoggingCallbackStepState(
            self.step + 1,
            episode_return,
            episode_length,
            done,
            average_return,
            average_length,
        )


def _restore_callback_scalars[T](pytree: T) -> T:
    """Restore Python scalar types that ``jax.debug.callback`` converts to arrays.

    When pytrees pass through ``jax.debug.callback``, Python ``bool`` and ``int``
    leaves are silently promoted to 0-d JAX arrays.  If those leaves are later
    traced by ``filter_jit`` they become abstract tracers, breaking any code that
    uses them in Python control flow (``if``, ``for length=``, etc.).

    This helper walks a pytree and converts every 0-d boolean or integer JAX
    array back to the corresponding Python scalar so that ``filter_jit``
    correctly places them in the static partition.
    """

    def _maybe_restore(leaf: Any) -> Any:
        if isinstance(leaf, jax.Array) and leaf.ndim == 0:
            if jnp.issubdtype(leaf.dtype, jnp.bool_):
                return bool(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.integer):
                return int(leaf)
        return leaf

    return jax.tree.map(_maybe_restore, pytree)


def _make_video_recorder(
    video_interval: int,
    video_num_steps: int,
    video_width: int,
    video_height: int,
    video_fps: float,
    backend: AbstractLoggingBackend,
    executor: concurrent.futures.ThreadPoolExecutor,
) -> Callable[..., None]:
    """
    Create a JIT-safe video recording callback.

    Submits an eval rollout to a background thread every ``video_interval``
    iterations. The callback itself returns immediately so JAX dispatch is not
    blocked; rendering and encoding happen concurrently with training.

    The rollout is compiled with ``lax.scan`` so the full ``video_num_steps``
    computation runs as a single XLA dispatch rather than a Python loop.

    Args:
        video_interval: Record a video every this many iterations.
        video_num_steps: Number of environment steps per video.
        video_width: Render width in pixels.
        video_height: Render height in pixels.
        video_fps: Playback frames per second.
        backend: Logging backend to forward video frames to.
        executor: Thread pool to run the recording work in.
    """

    @eqx.filter_jit
    def run_rollout(
        env: AbstractEnvLike,
        policy: AbstractPolicy,
        init_key: Key[Array, ""],
        policy_key: Key[Array, ""],
        rollout_key: Key[Array, ""],
    ):
        env_state = env.initial(key=init_key)
        policy_state = policy.reset(key=policy_key)

        def step_fn(carry, _):
            env_state, policy_state, rollout_key = carry
            obs_key, mask_key, step_key, reset_key, rollout_key = jr.split(
                rollout_key, 5
            )
            observation = env.observation(env_state, key=obs_key)
            action_mask = env.action_mask(env_state, key=mask_key)
            policy_state, action = policy(
                policy_state, observation, key=step_key, action_mask=action_mask
            )
            new_env_state = env.transition(env_state, action, key=step_key)
            done = env.terminal(new_env_state, key=step_key) | env.truncate(
                new_env_state
            )
            carry_env = filter_cond(
                done, lambda: env.initial(key=reset_key), lambda: new_env_state
            )
            carry_policy = filter_cond(
                done, lambda: policy.reset(key=reset_key), lambda: policy_state
            )
            return (carry_env, carry_policy, rollout_key), new_env_state

        _, env_states = filter_scan(
            step_fn,
            (env_state, policy_state, rollout_key),
            None,
            length=video_num_steps,
        )
        return env_states

    def _do_record(
        env: AbstractEnvLike,
        policy: AbstractPolicy,
        step: int,
        key: Key[Array, ""],
    ) -> None:
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
                    mujoco_renderer.render(env_state.unwrapped.sim_state)
                    rgb, _ = mujoco_renderer.read_pixels()
                    return rgb

            else:
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
            env = _restore_callback_scalars(env)
            policy = _restore_callback_scalars(policy)
            env_states = run_rollout(env, policy, init_key, policy_key, rollout_key)
            env_states = jax.device_get(env_states)

            frames = [
                render_frame(jax.tree.map(lambda x: x[i], env_states))
                for i in range(video_num_steps)
            ]

            renderer.close()

            frames_arr = np.stack(frames).astype(np.uint8)
            backend.log_video("eval/video", frames_arr, step, fps=video_fps)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Video recording failed at step {step}: {exc}",
                stacklevel=2,
            )

    @partial(callback_wrapper, ordered=True)
    def record_video(
        env: AbstractEnvLike,
        policy: AbstractPolicy,
        step: Int[Array, ""],
        iteration_count: Int[Array, ""],
        key: Key[Array, ""],
    ) -> None:
        if int(iteration_count) % video_interval != 0:
            return
        executor.submit(_do_record, env, policy, int(step), key)

    return record_video


class LoggingCallback(AbstractCallback[EmptyCallbackState, LoggingCallbackStepState]):
    """Callback that logs training metrics to a pluggable logging backend.

    The backend is opened immediately at construction, so a single
    ``LoggingCallback`` instance can be reused across multiple
    ``learn()`` calls and all metrics will be logged to the same run.
    Call ``close()`` when finished to flush data and release resources.

    At the end of each iteration the following metrics are logged:

    - ``episode/return``: EMA of per-episode returns across all environments.
    - ``episode/length``: EMA of per-episode lengths.
    - ``train/*``: All entries in the algorithm ``training_log``.
    - ``train/learning_rate``: Current learning rate from the Optax state
      (``NaN`` when not available).

    When ``video_interval`` is positive an evaluation rollout is recorded
    every ``video_interval`` iterations and forwarded to the backend as
    ``eval/video``.

    At each training start, hyperparameters are logged via ``log_hparams``.
    Policy scalar fields (prefixed ``policy.``) and algorithm scalar fields
    (prefixed ``algorithm.``) are extracted automatically; any extra entries
    provided in ``hparams`` are merged on top (explicit values take
    precedence).

    Note:
        This callback must be constructed **outside** any JIT-compiled
        function.

    Attributes:
        alpha: EMA smoothing factor for episode statistics.

    Args:
        backend: Logging backend to send metrics to.
        name: Explicit run name. When ``None``, a name is generated from the
            environment name, policy name, and a timestamp. If neither ``env``
            nor ``policy`` are provided, falls back to a plain timestamp.
        env: Environment used to derive the run name when ``name`` is ``None``.
        policy: Policy used to derive the run name when ``name`` is ``None``.
        alpha: EMA smoothing factor (higher = more weight on recent episodes).
        hparams: Additional explicit hyperparameters. Merged last so these
            values take precedence over auto-extracted ones.
        video_interval: Record video every this many iterations; ``0`` disables.
        video_num_steps: Environment steps per recorded video.
        video_width: Render width in pixels.
        video_height: Render height in pixels.
        video_fps: Playback frames per second.
    """

    _backend: AbstractLoggingBackend = eqx.field(static=True)
    _name: str | None = eqx.field(static=True)
    _hparams: dict[str, Any] | None = eqx.field(static=True)
    alpha: float
    _record_video_fn: Callable[..., None] | None = eqx.field(static=True)
    _video_executor: concurrent.futures.ThreadPoolExecutor | None = eqx.field(
        static=True
    )

    def __init__(
        self,
        backend: AbstractLoggingBackend,
        name: str | None = None,
        env: AbstractEnvLike | None = None,
        policy: AbstractPolicy | None = None,
        alpha: float = 0.9,
        hparams: dict[str, Any] | None = None,
        video_interval: int = 0,
        video_num_steps: int = 128,
        video_width: int = 640,
        video_height: int = 480,
        video_fps: float = 50.0,
    ) -> None:
        self._backend = backend
        self._hparams = hparams
        self.alpha = alpha

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts = []
            if policy is not None:
                parts.append(policy.name)
            if env is not None:
                parts.append(env.name)
            parts.append(timestamp)
            name = "_".join(parts)
        self._name = name

        backend.open(name)

        if video_interval > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._video_executor = executor
            self._record_video_fn = _make_video_recorder(
                video_interval,
                video_num_steps,
                video_width,
                video_height,
                video_fps,
                backend,
                executor,
            )
        else:
            self._video_executor = None
            self._record_video_fn = None

    def reset(self, ctx: ResetContext, *, key: Key[Array, ""]) -> EmptyCallbackState:
        return EmptyCallbackState()

    def step_reset(
        self, ctx: ResetContext, *, key: Key[Array, ""]
    ) -> LoggingCallbackStepState:
        return LoggingCallbackStepState.initial()

    def on_step(
        self, ctx: StepContext, *, key: Key[Array, ""]
    ) -> LoggingCallbackStepState:
        return ctx.state.next(ctx.reward, ctx.done, self.alpha)

    def on_iteration(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        log = ctx.training_log
        opt_state = ctx.opt_state

        learning_rate = optax.tree_utils.tree_get(
            opt_state,
            "learning_rate",
            jnp.nan,
            filtering=lambda _, value: isinstance(value, jnp.ndarray),
        )

        step_state = ctx.step_state
        last_step = step_state.step.sum()

        scalars = {f"train/{k}": v for k, v in log.items()}
        scalars["train/learning_rate"] = learning_rate
        scalars["episode/return"] = step_state.average_return.mean()
        scalars["episode/length"] = step_state.average_length.mean()

        callback_with_numpy_wrapper(self._backend.log_scalars, ordered=True)(
            scalars, last_step
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
        hparams: dict[str, Any] = {}
        hparams.update(
            {f"policy.{k}": v for k, v in _extract_hparams(ctx.policy).items()}
        )
        hparams.update(
            {f"algorithm.{k}": v for k, v in _extract_hparams(ctx.algorithm).items()}
        )
        hparams.update(self._hparams or {})

        callback_wrapper(lambda: self._backend.log_hparams(hparams), ordered=True)()
        return ctx.state

    def on_training_end(
        self, ctx: TrainingContext, *, key: Key[Array, ""]
    ) -> EmptyCallbackState:
        return ctx.state

    def close(self) -> None:
        """Flush pending data and release backend resources.

        Call this after all ``learn()`` calls are complete. The backend
        remains open between ``learn()`` calls so that metrics from
        multiple stages are logged to the same run.
        """
        if self._video_executor is not None:
            self._video_executor.shutdown(wait=True)
        self._backend.close()

    def continue_training(
        self, ctx: IterationContext, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return jnp.array(True)
