from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax import random as jr
from jaxtyping import Array, Float, Int

from oryx.env import AbstractEnv
from oryx.spaces import Box
from oryx.wrappers import (
    ClipAction,
    EpisodeStatistics,
    RescaleAction,
    TransformAction,
)
from oryx.wrappers.utils import rescale_box


class EchoEnv(AbstractEnv[Float[Array, " n"], Float[Array, " n"]]):
    """
    Echoes the action back as the observation, never terminates.
    """

    state_index: eqx.nn.StateIndex[None]
    _action_space: Box
    _observation_space: Box

    def __init__(
        self,
        action_space: Box = Box(-jnp.inf, jnp.inf),
        observation_space: Box = Box(-jnp.inf, jnp.inf),
    ):
        self._action_space = action_space
        self._observation_space = observation_space
        self.state_index = eqx.nn.StateIndex(None)

    def reset(self, state, *, key):
        return state, self.observation_space.sample(key), {}

    def step(self, state, action, *, key):
        return (
            state,
            action,
            jnp.asarray(0.0),
            jnp.asarray(False),
            jnp.asarray(False),
            {},
        )

    def render(self, state):
        pass

    def close(self):
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space


class FiniteEpisodeEnv(AbstractEnv[Float[Array, ""], Float[Array, ""]]):
    """
    Counts steps and terminates after ``done_at`` steps.
    """

    state_index: eqx.nn.StateIndex[Int[Array, ""]]
    done_at: int

    def __init__(self, *, key, done_at: int = 3):
        self.state_index = eqx.nn.StateIndex(jnp.array(0))
        self.done_at = done_at

    def reset(self, state, *, key):
        state = state.set(self.state_index, jnp.array(0))
        return state, jnp.asarray(0.0), {}

    def step(self, state, action, *, key):
        step_no = state.get(self.state_index) + 1
        state = state.set(self.state_index, step_no)

        termination = step_no == self.done_at
        return (
            state,
            step_no,
            jnp.asarray(1.0),
            termination,
            jnp.asarray(False),
            {},
        )

    def render(self, state):
        pass

    def close(self):
        pass

    @property
    def action_space(self):
        return Box(-jnp.inf, jnp.inf)

    @property
    def observation_space(self):
        return Box(-jnp.inf, jnp.inf)


def test_transform_action_wrapper_passes_func():
    """The wrapper must forward `func(action)` to the env."""
    reset_key, step_key = jr.split(jr.key(0), 2)

    action = jnp.array(1.0)

    def func(a):
        return 2 * a

    env, state = eqx.nn.make_with_state(EchoEnv)()
    wrapper = TransformAction(
        env=env,
        func=func,
        action_space=Box(-jnp.inf, jnp.inf),
    )

    state, obs, *_ = wrapper.reset(state, key=reset_key)
    state, obs, *_ = wrapper.step(state, action, key=step_key)

    assert jnp.allclose(obs, func(action)), "Env should receive transformed action"


@pytest.mark.parametrize("low, high", [([-1.0, -1.0], [1.0, 1.0])])
def test_clip_action_wrapper_clips_values_and_space(low, high):
    low, high = jnp.asarray(low), jnp.asarray(high)
    reset_key, step_key = jr.split(jr.key(1), 2)

    env, state = eqx.nn.make_with_state(EchoEnv)(action_space=Box(low, high))
    wrapper = ClipAction(env=env)

    over_action = jnp.array(high) + 1
    expected = jnp.clip(over_action, low, high)

    state, _, _ = wrapper.reset(state, key=reset_key)
    state, obs, *_ = wrapper.step(state, over_action, key=step_key)

    assert jnp.allclose(obs, expected), "Action should be clipped before reaching env"
    assert jnp.all(jnp.isinf(wrapper.action_space.low)) and jnp.all(
        jnp.isinf(wrapper.action_space.high)
    ), "Wrapper advertises an unbounded action space"


@pytest.mark.parametrize(
    "action, expected",
    [(-1.0, 0.0), (0.0, 5.0), (1.0, 10.0)],
)
def test_rescale_action_wrapper_affine(action, expected):
    key = jr.key(2)
    reset_key, step_key = jr.split(key, 2)

    env, state = eqx.nn.make_with_state(EchoEnv)(
        action_space=Box(0.0, 10.0), observation_space=Box(0.0, 10.0)
    )
    wrapper = RescaleAction(env=env)

    state, _, _ = wrapper.reset(state, key=reset_key)
    state, obs, *_ = wrapper.step(state, action, key=step_key)

    print(f"Action: {action}, Expected: {expected}, Obs: {obs}")

    assert jnp.allclose(
        obs, expected
    ), "Affine rescaling should map [-1,0,1] to [low,mid,high]"

    assert jnp.allclose(wrapper.action_space.low, -1.0) and jnp.allclose(
        wrapper.action_space.high, 1.0
    )


def test_rescale_box_roundtrip():
    """Utility functions must invert each other."""
    orig = Box(0.0, 10.0)
    _, forward, backward = rescale_box(orig, -1.0, 1.0)

    sample = orig.sample(jr.key(3))

    assert jnp.allclose(
        backward(forward(sample)), sample
    ), "forward∘backward should be identity"


def test_episode_statistics_wrapper_reset_counters():
    """After reset, stats must be zeroed."""
    env_key, reset_key = jr.split(jr.key(4), 2)
    env = FiniteEpisodeEnv(key=env_key)
    wrapper, state = eqx.nn.make_with_state(EpisodeStatistics)(env=env)

    state, _, info = wrapper.reset(state, key=reset_key)

    ep = info["episode"]
    assert (
        ep["length"] == 0 and ep["reward"] == 0 and not bool(ep["done"])
    ), "Counters should start at zero"


def test_episode_statistics_wrapper_accumulates_scan():
    """Counters grow until termination flag is raised (scan version)."""
    env_key, reset_key, key0 = jr.split(jr.key(5), 3)
    env = FiniteEpisodeEnv(key=env_key)

    wrapper, state = eqx.nn.make_with_state(EpisodeStatistics)(env=env)
    state, _, _ = wrapper.reset(state, key=reset_key)

    def step_fn(carry, _):
        state, key = carry
        key, step_key = jr.split(key)

        state, *_, info = wrapper.step(state, jnp.array(0.0), key=step_key)
        carry = (state, key)
        return carry, info

    _, infos = lax.scan(
        step_fn,
        (state, key0),
        None,
        length=env.done_at,
    )

    last_info = jax.tree.map(lambda x: x[-1], infos)
    ep = last_info["episode"]
    assert (
        ep["length"] == 3 and ep["reward"] == 3 and bool(ep["done"])
    ), "Episode stats should reflect accumulated steps/rewards and signal done"


def test_nested_wrapper_unwrapped_returns_base():
    base_env, _ = eqx.nn.make_with_state(EchoEnv)()

    clip_env = ClipAction(env=base_env)

    outer = EpisodeStatistics(env=clip_env)

    assert (
        outer.unwrapped is base_env
    ), "unwrapped must return the original env instance"
