from __future__ import annotations

import equinox as eqx
import jax
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.spaces import Box
from oryx.wrappers import (
    ClipAction,
    ClipObservation,
    ClipReward,
    EpisodeStatistics,
    FlattenObservation,
    Identity,
    RescaleAction,
    RescaleObservation,
    TimeLimit,
    TransformAction,
)
from oryx.wrappers.utils import rescale_box

from .shared import EchoEnv, FiniteEpisodeEnv, PassThroughEnv


def test_nested_wrapper_unwrapped():
    env, _ = eqx.nn.make_with_state(EchoEnv)()

    wrapper = Identity(Identity(env))

    assert wrapper.unwrapped is env, "unwrapped must return the original env instance"


def test_transform_action():
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
def test_clip_action(low, high):
    low, high = jnp.asarray(low), jnp.asarray(high)
    reset_key, step_key = jr.split(jr.key(9), 2)

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
def test_rescale_action(action, expected):
    reset_key, step_key = jr.split(jr.key(0), 2)

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


def test_rescale_box():
    orig = Box(0.0, 10.0)
    _, forward, backward = rescale_box(orig, -1.0, 1.0)

    sample = orig.sample(jr.key(3))

    assert jnp.allclose(
        backward(forward(sample)), sample
    ), "forward∘backward should be identity"


def test_episode_statistics():
    env_key, reset_key, key0 = jr.split(jr.key(0), 3)
    env = FiniteEpisodeEnv(key=env_key)

    wrapper, state = eqx.nn.make_with_state(EpisodeStatistics)(env=env)
    state, _, info = wrapper.reset(state, key=reset_key)

    ep = info["episode"]
    assert (
        ep["length"] == 0 and ep["reward"] == 0 and not bool(ep["done"])
    ), "Counters should start at zero"

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


@pytest.mark.parametrize("low, high", [([-1.0, -1.0], [1.0, 1.0])])
def test_clip_observation(low, high):
    low, high = jnp.asarray(low), jnp.asarray(high)
    over_val = jnp.array(high) + 2.0

    reset_key, step_key = jr.split(jr.key(0), 2)

    base_env, state = eqx.nn.make_with_state(EchoEnv)(
        action_space=Box(low, high),
        observation_space=Box(low, high),
    )
    wrapper = ClipObservation(env=base_env)

    state, _, _ = wrapper.reset(state, key=reset_key)
    state, obs, *_ = wrapper.step(state, over_val, key=step_key)

    assert jnp.allclose(
        obs, jnp.clip(over_val, low, high)
    ), "Observation should be element-wise clipped"

    assert wrapper.observation_space == base_env.observation_space


@pytest.mark.parametrize("raw, expected", [(0.0, -1.0), (5.0, 0.0), (10.0, 1.0)])
def test_rescale_observation(raw, expected):
    reset_key, step_key = jr.split(jr.key(0), 2)

    base_env, state = eqx.nn.make_with_state(EchoEnv)(
        action_space=Box(0.0, 10.0),
        observation_space=Box(0.0, 10.0),
    )
    wrapper = RescaleObservation(env=base_env)

    state, _, _ = wrapper.reset(state, key=reset_key)
    state, obs, *_ = wrapper.step(state, jnp.asarray(raw), key=step_key)

    assert pytest.approx(float(obs)) == expected

    assert jnp.allclose(wrapper.observation_space.low, -1.0)
    assert jnp.allclose(wrapper.observation_space.high, 1.0)


@pytest.mark.parametrize(
    "action, min_, max_, expected",
    [
        (-5.0, -1.0, 1.0, -1.0),
        (0.0, -1.0, 1.0, 0.0),
        (7.5, -1.0, 1.0, 1.0),
    ],
)
def test_clip_reward(action, min_, max_, expected):
    reset_key, step_key = jr.split(jr.key(0), 2)

    base_env, state = eqx.nn.make_with_state(PassThroughEnv)()
    wrapper = ClipReward(env=base_env, min=min_, max=max_)

    state, _, _ = wrapper.reset(state, key=reset_key)
    state, _, reward, *_ = wrapper.step(state, jnp.asarray(action), key=step_key)

    assert pytest.approx(float(reward)) == expected, "Reward not correctly clipped"


def test_time_limit(steps=2):
    reset_key, step_key = jr.split(jr.key(0), 2)

    env = EchoEnv()
    wrapper, state = eqx.nn.make_with_state(TimeLimit)(env=env, max_episode_steps=steps)

    state, _, _ = wrapper.reset(state, key=reset_key)

    for i in range(1, steps * 2 + 1):
        state, _, _, _, truncation, _ = wrapper.step(
            state, jnp.asarray(0.0), key=step_key
        )

        if i % steps == 0:
            assert truncation, "Should truncate after max_episode_steps"
        else:
            assert not truncation, "Should not truncate before max_episode_steps"

        if truncation:
            state, _, _ = wrapper.reset(state, key=reset_key)


def test_flatten_observation():
    reset_key, step_key = jr.split(jr.key(0), 2)

    obs_shape = (2, 3)
    base_env, state = eqx.nn.make_with_state(EchoEnv)(
        action_space=Box(-jnp.inf, jnp.inf, shape=obs_shape),
        observation_space=Box(-jnp.inf, jnp.inf, shape=obs_shape),
    )
    wrapper = FlattenObservation(env=base_env)

    expected_flat_dim = int(jnp.prod(jnp.asarray(obs_shape)))
    assert wrapper.observation_space == Box(
        -jnp.inf, jnp.inf, shape=(expected_flat_dim,)
    )

    state, obs, _ = wrapper.reset(state, key=reset_key)
    assert obs.shape == (expected_flat_dim,)

    action = jnp.arange(expected_flat_dim).reshape(obs_shape)
    state, obs_step, *_ = wrapper.step(state, action, key=step_key)

    assert obs_step.shape == (expected_flat_dim,)
    assert jnp.array_equal(obs_step, action.ravel()), "Flattening mismatch"
