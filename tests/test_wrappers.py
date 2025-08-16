from __future__ import annotations

import equinox as eqx
import jax
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.space import Box
from oryx.wrapper import (
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
from oryx.wrapper.utils import rescale_box

from .shared import EchoEnv, FiniteEpisodeEnv, PassThroughEnv


def test_nested_wrapper_unwrapped():
    scan_key, reset_key = jr.split(jr.key(0), 2)

    env, state = eqx.nn.make_with_state(EchoEnv)()
    wrapper = Identity(Identity(env))

    assert wrapper.unwrapped is env, "unwrapped must return the original env instance"

    state, _, _ = wrapper.reset(state, key=reset_key)

    action = jnp.asarray(1.23)

    def step_fn(carry, _):
        st, k = carry
        k, sk = jr.split(k)
        st, obs, *_ = wrapper.step(st, action, key=sk)
        return (st, k), obs

    (_, _), obs_seq = lax.scan(step_fn, (state, scan_key), None, length=5)
    assert jnp.allclose(obs_seq, action)


def test_transform_action():
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)

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

    def step_scan(carry, _):
        st, k = carry
        k, sk = jr.split(k)
        st, ob, *_ = wrapper.step(st, action, key=sk)
        return (st, k), ob

    (_, _), obs_seq = lax.scan(step_scan, (state, scan_key), None, length=4)
    assert jnp.allclose(obs_seq, func(action))


@pytest.mark.parametrize("low, high", [([-1.0, -1.0], [1.0, 1.0])])
def test_clip_action(low, high):
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
    low, high = jnp.asarray(low), jnp.asarray(high)

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

    def step_scan(carry, _):
        st, k = carry
        k, sk = jr.split(k)
        st, ob, *_ = wrapper.step(st, over_action, key=sk)
        return (st, k), ob

    (_, _), obs_seq = lax.scan(step_scan, (state, scan_key), None, length=3)
    assert jnp.allclose(obs_seq, expected)


@pytest.mark.parametrize(
    "action, expected",
    [(-1.0, 0.0), (0.0, 5.0), (1.0, 10.0)],
)
def test_rescale_action(action, expected):
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)

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

    actions = jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected_seq = (actions + 1.0) * 5.0

    def step_scan(carry, a):
        st, k = carry
        k, sk = jr.split(k)
        st, ob, *_ = wrapper.step(st, a, key=sk)
        return (st, k), ob

    (_, _), obs_seq = lax.scan(step_scan, (state, scan_key), actions)
    assert jnp.allclose(obs_seq, expected_seq)


def test_rescale_box():
    scan_key, sample_key = jr.split(jr.key(3), 2)

    orig = Box(0.0, 10.0)
    _, forward, backward = rescale_box(orig, -1.0, 1.0)

    sample = orig.sample(sample_key)

    assert jnp.allclose(
        backward(forward(sample)), sample
    ), "forward∘backward should be identity"

    keys = jr.split(scan_key, 6)

    def gen_and_roundtrip(carry, k):
        s = orig.sample(k)
        r = backward(forward(s))
        return carry, (s, r)

    _, (s_seq, r_seq) = lax.scan(gen_and_roundtrip, None, keys)
    assert jnp.allclose(s_seq, r_seq)


def test_episode_statistics():
    scan_key, env_key, reset_key = jr.split(jr.key(0), 3)
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
        (state, scan_key),
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
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
    low, high = jnp.asarray(low), jnp.asarray(high)
    over_val = jnp.array(high) + 2.0

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

    def step_scan(carry, _):
        st, k = carry
        k, sk = jr.split(k)
        st, ob, *_ = wrapper.step(st, over_val, key=sk)
        return (st, k), ob

    (_, _), obs_seq = lax.scan(step_scan, (state, scan_key), None, length=4)
    assert jnp.all(obs_seq <= high) and jnp.all(obs_seq >= low)


@pytest.mark.parametrize("raw, expected", [(0.0, -1.0), (5.0, 0.0), (10.0, 1.0)])
def test_rescale_observation(raw, expected):
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)

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

    raws = jnp.asarray([0.0, 2.5, 5.0, 7.5, 10.0])
    expected_seq = (raws - 5.0) / 5.0

    def step_scan(carry, x):
        st, k = carry
        k, sk = jr.split(k)
        st, ob, *_ = wrapper.step(st, x, key=sk)
        return (st, k), ob

    (_, _), obs_seq = lax.scan(step_scan, (state, scan_key), raws)
    assert jnp.allclose(obs_seq, expected_seq, atol=1e-7)


@pytest.mark.parametrize(
    "action, min, max, expected",
    [
        (-5.0, -1.0, 1.0, -1.0),
        (0.0, -1.0, 1.0, 0.0),
        (7.5, -1.0, 1.0, 1.0),
    ],
)
def test_clip_reward(action, min, max, expected):
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)

    base_env, state = eqx.nn.make_with_state(PassThroughEnv)()
    wrapper = ClipReward(env=base_env, min=min, max=max)

    state, _, _ = wrapper.reset(state, key=reset_key)
    state, _, reward, *_ = wrapper.step(state, jnp.asarray(action), key=step_key)

    assert pytest.approx(float(reward)) == expected, "Reward not correctly clipped"

    acts = jnp.asarray([-5.0, -0.1, 0.0, 0.3, 7.5])
    exp_seq = jnp.clip(acts, min=min, max=max)

    def step_scan(carry, a):
        st, k = carry
        k, sk = jr.split(k)
        st, _, r, *_ = wrapper.step(st, a, key=sk)
        return (st, k), r

    (_, _), rew_seq = lax.scan(step_scan, (state, scan_key), acts)
    assert jnp.allclose(rew_seq, exp_seq)


def test_time_limit(steps=2):
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)

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

    state, _, _ = wrapper.reset(state, key=reset_key)

    def step_scan(carry, _):
        st, k = carry
        st, _, _, _, trunc, _ = wrapper.step(st, jnp.asarray(0.0), key=k)
        return (st, k), trunc

    (_, _), trunc_seq = lax.scan(step_scan, (state, scan_key), None, length=steps)
    assert not bool(trunc_seq[:-1].any())
    assert bool(trunc_seq[-1])


def test_flatten_observation():
    scan_key, reset_key, step_key = jr.split(jr.key(0), 3)

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

    actions = jnp.stack(
        [
            jnp.arange(expected_flat_dim).reshape(obs_shape),
            jnp.ones(obs_shape),
            jnp.zeros(obs_shape),
        ],
        axis=0,
    )

    def step_scan(carry, a):
        st, k = carry
        k, sk = jr.split(k)
        st, ob, *_ = wrapper.step(st, a, key=sk)
        return (st, k), ob

    (_, _), obs_seq = lax.scan(step_scan, (state, scan_key), actions)
    assert obs_seq.shape == (actions.shape[0], expected_flat_dim)
    assert jnp.array_equal(obs_seq[0], actions[0].ravel())
