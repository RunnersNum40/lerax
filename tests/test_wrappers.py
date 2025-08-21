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

from .shared.envs import EchoEnv, FiniteEpisodeEnv, PassThroughEnv


class TestIdentity:
    def test_interface(self):
        scan_key, reset_key = jr.split(jr.key(0), 2)
        env, state = eqx.nn.make_with_state(EchoEnv)()
        w = Identity(Identity(env))
        assert w.unwrapped is env

        state, _, _ = w.reset(state, key=reset_key)
        action = jnp.asarray(1.23)

        def step_fn(carry, _):
            st, k = carry
            k, sk = jr.split(k)
            st, obs, *_ = w.step(st, action, key=sk)
            return (st, k), obs

        (_, _), obs_seq = lax.scan(step_fn, (state, scan_key), None, length=5)
        assert jnp.allclose(obs_seq, action)

    def test_jit(self):
        key = jr.key(0)
        env, state = eqx.nn.make_with_state(EchoEnv)()
        w = Identity(env)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            k1, k2 = jr.split(k)
            st, ob, *_ = ww.step(st, jnp.asarray(0.5), key=k2)
            return st, ob

        _, ob = f(w, state, key)
        assert ob.shape == ()


class TestTransformAction:
    def test_interface(self):
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        env, state = eqx.nn.make_with_state(EchoEnv)()
        func = lambda a: 2 * a
        w = TransformAction(env=env, func=func, action_space=Box(-jnp.inf, jnp.inf))

        state, _, _ = w.reset(state, key=reset_key)
        state, obs, *_ = w.step(state, jnp.asarray(1.0), key=step_key)
        assert jnp.allclose(obs, 2.0)

        def step_scan(carry, _):
            st, k = carry
            k, sk = jr.split(k)
            st, ob, *_ = w.step(st, jnp.asarray(1.0), key=sk)
            return (st, k), ob

        (_, _), obs_seq = lax.scan(step_scan, (state, scan_key), None, length=4)
        assert jnp.allclose(obs_seq, 2.0)

    def test_jit(self):
        key = jr.key(0)
        env, state = eqx.nn.make_with_state(EchoEnv)()
        w = TransformAction(
            env=env, func=lambda a: a + 3, action_space=Box(-jnp.inf, jnp.inf)
        )

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, ob, *_ = ww.step(st, jnp.asarray(2.0), key=k)
            return ob

        ob = f(w, state, key)
        assert float(ob) == pytest.approx(5.0)


class TestClipAction:
    @pytest.mark.parametrize("low, high", [([-1.0, -1.0], [1.0, 1.0])])
    def test_interface(self, low, high):
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        low, high = jnp.asarray(low), jnp.asarray(high)
        env, state = eqx.nn.make_with_state(EchoEnv)(space=Box(low, high))
        w = ClipAction(env=env)

        over = high + 1
        state, _, _ = w.reset(state, key=reset_key)
        state, obs, *_ = w.step(state, over, key=step_key)
        assert jnp.allclose(obs, jnp.clip(over, low, high))
        assert jnp.all(jnp.isinf(w.action_space.low)) and jnp.all(
            jnp.isinf(w.action_space.high)
        )

        def scan(carry, _):
            st, k = carry
            k, sk = jr.split(k)
            st, ob, *_ = w.step(st, over, key=sk)
            return (st, k), ob

        (_, _), obs_seq = lax.scan(scan, (state, scan_key), None, length=3)
        assert jnp.allclose(obs_seq, jnp.clip(over, low, high))

    def test_jit(self):
        key = jr.key(0)
        env, state = eqx.nn.make_with_state(EchoEnv)(
            space=Box(-jnp.ones(2), jnp.ones(2))
        )
        w = ClipAction(env=env)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, ob, *_ = ww.step(st, jnp.asarray([3.0, -5.0]), key=k)
            return ob

        ob = f(w, state, key)
        assert jnp.all(ob <= 1.0) and jnp.all(ob >= -1.0)


class TestRescaleAction:
    @pytest.mark.parametrize(
        "action, expected",
        [(-1.0, 0.0), (0.0, 5.0), (1.0, 10.0)],
    )
    def test_interface(self, action, expected):
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        env, state = eqx.nn.make_with_state(EchoEnv)(space=Box(0.0, 10.0))
        w = RescaleAction(env=env)

        state, _, _ = w.reset(state, key=reset_key)
        state, obs, *_ = w.step(state, jnp.asarray(action), key=step_key)
        assert pytest.approx(float(obs)) == expected
        assert jnp.allclose(w.action_space.low, -1.0) and jnp.allclose(
            w.action_space.high, 1.0
        )

        actions = jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
        expected_seq = (actions + 1.0) * 5.0

        def scan(carry, a):
            st, k = carry
            k, sk = jr.split(k)
            st, ob, *_ = w.step(st, a, key=sk)
            return (st, k), ob

        (_, _), obs_seq = lax.scan(scan, (state, scan_key), actions)
        assert jnp.allclose(obs_seq, expected_seq)

    def test_jit(self):
        key = jr.key(0)
        env, state = eqx.nn.make_with_state(EchoEnv)(space=Box(2.0, 4.0))
        w = RescaleAction(env=env)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, ob, *_ = ww.step(st, jnp.asarray(0.0), key=k)
            return ob

        ob = f(w, state, key)
        assert float(ob) == pytest.approx(3.0)


class TestRescaleBox:
    def test_roundtrip(self):
        scan_key, sample_key = jr.split(jr.key(0), 2)
        orig = Box(0.0, 10.0)
        _, forward, backward = rescale_box(orig, -1.0, 1.0)
        sample = orig.sample(sample_key)
        assert jnp.allclose(backward(forward(sample)), sample)

        keys = jr.split(scan_key, 6)

        def gen_and_roundtrip(carry, k):
            s = orig.sample(k)
            r = backward(forward(s))
            return carry, (s, r)

        _, (s_seq, r_seq) = lax.scan(gen_and_roundtrip, None, keys)
        assert jnp.allclose(s_seq, r_seq)


class TestEpisodeStatistics:
    def test_interface(self):
        scan_key, env_key, reset_key = jr.split(jr.key(0), 3)
        env = FiniteEpisodeEnv(key=env_key)
        w, state = eqx.nn.make_with_state(EpisodeStatistics)(env=env)

        state, _, info = w.reset(state, key=reset_key)
        ep = info["episode"]
        assert ep["length"] == 0 and ep["reward"] == 0 and not bool(ep["done"])

        def step_fn(carry, _):
            st, k = carry
            k, sk = jr.split(k)
            st, *_, info = w.step(st, jnp.asarray(0.0), key=sk)
            return (st, k), info

        _, infos = lax.scan(step_fn, (state, scan_key), None, length=env.done_at)
        last_info = jax.tree.map(lambda x: x[-1], infos)
        ep = last_info["episode"]
        assert ep["length"] == 3 and ep["reward"] == 3 and bool(ep["done"])

    def test_jit(self):
        key = jr.key(0)
        env = FiniteEpisodeEnv(key=key)
        w, state = eqx.nn.make_with_state(EpisodeStatistics)(env=env)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, _, _, _, _, info = ww.step(st, jnp.asarray(0.0), key=k)
            return info["episode"]["length"]

        length = f(w, state, key)
        assert int(length) in (0, 1)


class TestClipObservation:
    @pytest.mark.parametrize("low, high", [([-1.0, -1.0], [1.0, 1.0])])
    def test_interface(self, low, high):
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        low, high = jnp.asarray(low), jnp.asarray(high)
        over = jnp.array(high) + 2.0

        base_env, state = eqx.nn.make_with_state(EchoEnv)(space=Box(low, high))
        w = ClipObservation(env=base_env)

        state, _, _ = w.reset(state, key=reset_key)
        state, obs, *_ = w.step(state, over, key=step_key)
        assert jnp.allclose(obs, jnp.clip(over, low, high))
        assert w.observation_space == base_env.observation_space

        def scan(carry, _):
            st, k = carry
            k, sk = jr.split(k)
            st, ob, *_ = w.step(st, over, key=sk)
            return (st, k), ob

        (_, _), obs_seq = lax.scan(scan, (state, scan_key), None, length=4)
        assert jnp.all(obs_seq <= high) and jnp.all(obs_seq >= low)

    def test_jit(self):
        key = jr.key(0)
        base_env, state = eqx.nn.make_with_state(EchoEnv)(
            space=Box(-jnp.ones(2), jnp.ones(2))
        )
        w = ClipObservation(env=base_env)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, ob, *_ = ww.step(st, jnp.asarray([5.0, -5.0]), key=k)
            return ob

        ob = f(w, state, key)
        assert jnp.all(ob <= 1.0) and jnp.all(ob >= -1.0)


class TestRescaleObservation:
    @pytest.mark.parametrize("raw, expected", [(0.0, -1.0), (5.0, 0.0), (10.0, 1.0)])
    def test_interface(self, raw, expected):
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        base_env, state = eqx.nn.make_with_state(EchoEnv)(space=Box(0.0, 10.0))
        w = RescaleObservation(env=base_env)

        state, _, _ = w.reset(state, key=reset_key)
        state, obs, *_ = w.step(state, jnp.asarray(raw), key=step_key)
        assert pytest.approx(float(obs)) == expected
        assert jnp.allclose(w.observation_space.low, -1.0)
        assert jnp.allclose(w.observation_space.high, 1.0)

        raws = jnp.asarray([0.0, 2.5, 5.0, 7.5, 10.0])
        expected_seq = (raws - 5.0) / 5.0

        def scan(carry, x):
            st, k = carry
            k, sk = jr.split(k)
            st, ob, *_ = w.step(st, x, key=sk)
            return (st, k), ob

        (_, _), obs_seq = lax.scan(scan, (state, scan_key), raws)
        assert jnp.allclose(obs_seq, expected_seq, atol=1e-7)

    def test_jit(self):
        key = jr.key(0)
        base_env, state = eqx.nn.make_with_state(EchoEnv)(space=Box(0.0, 2.0))
        w = RescaleObservation(env=base_env)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, ob, *_ = ww.step(st, jnp.asarray(1.0), key=k)
            return ob

        ob = f(w, state, key)
        assert float(ob) == pytest.approx(0.0)


class TestClipReward:
    @pytest.mark.parametrize(
        "action, minv, maxv, expected",
        [(-5.0, -1.0, 1.0, -1.0), (0.0, -1.0, 1.0, 0.0), (7.5, -1.0, 1.0, 1.0)],
    )
    def test_interface(self, action, minv, maxv, expected):
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        base_env, state = eqx.nn.make_with_state(PassThroughEnv)()
        w = ClipReward(env=base_env, min=minv, max=maxv)

        state, _, _ = w.reset(state, key=reset_key)
        state, _, reward, *_ = w.step(state, jnp.asarray(action), key=step_key)
        assert pytest.approx(float(reward)) == expected

        acts = jnp.asarray([-5.0, -0.1, 0.0, 0.3, 7.5])
        exp_seq = jnp.clip(acts, min=minv, max=maxv)

        def scan(carry, a):
            st, k = carry
            k, sk = jr.split(k)
            st, _, r, *_ = w.step(st, a, key=sk)
            return (st, k), r

        (_, _), rew_seq = lax.scan(scan, (state, scan_key), acts)
        assert jnp.allclose(rew_seq, exp_seq)

    def test_jit(self):
        key = jr.key(0)
        base_env, state = eqx.nn.make_with_state(PassThroughEnv)()
        w = ClipReward(env=base_env, min=-1.0, max=1.0)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, _, r, *_ = ww.step(st, jnp.asarray(3.0), key=k)
            return r

        r = f(w, state, key)
        assert float(r) == pytest.approx(1.0)


class TestTimeLimit:
    def test_interface(self):
        steps = 2
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        env = EchoEnv()
        w, state = eqx.nn.make_with_state(TimeLimit)(env=env, max_episode_steps=steps)

        state, _, _ = w.reset(state, key=reset_key)

        for i in range(1, steps * 2 + 1):
            state, _, _, _, trunc, _ = w.step(state, jnp.asarray(0.0), key=step_key)
            if i % steps == 0:
                assert trunc
                state, _, _ = w.reset(state, key=reset_key)
            else:
                assert not trunc

        state, _, _ = w.reset(state, key=reset_key)

        def scan(carry, _):
            st, k = carry
            st, _, _, _, trunc, _ = w.step(st, jnp.asarray(0.0), key=k)
            return (st, k), trunc

        (_, _), trunc_seq = lax.scan(scan, (state, scan_key), None, length=steps)
        assert not bool(trunc_seq[:-1].any())
        assert bool(trunc_seq[-1])

    def test_jit(self):
        key = jr.key(0)
        w, state = eqx.nn.make_with_state(TimeLimit)(env=EchoEnv(), max_episode_steps=3)

        @eqx.filter_jit
        def f(ww, st, k):
            st, _, _ = ww.reset(st, key=k)
            st, _, _, _, trunc, _ = ww.step(st, jnp.asarray(0.0), key=k)
            return trunc

        trunc = f(w, state, key)
        assert trunc.shape == ()


class TestFlattenObservation:
    def test_interface(self):
        scan_key, reset_key, step_key = jr.split(jr.key(0), 3)
        obs_shape = (2, 3)
        base_env, state = eqx.nn.make_with_state(EchoEnv)(
            space=Box(-jnp.inf, jnp.inf, shape=obs_shape)
        )
        w = FlattenObservation(env=base_env)

        expected = int(jnp.prod(jnp.asarray(obs_shape)))
        assert w.observation_space == Box(-jnp.inf, jnp.inf, shape=(expected,))

        state, obs, _ = w.reset(state, key=reset_key)
        assert obs.shape == (expected,)

        action = jnp.arange(expected).reshape(obs_shape)
        state, obs_step, *_ = w.step(state, action, key=step_key)
        assert obs_step.shape == (expected,)
        assert jnp.array_equal(obs_step, action.ravel())

        actions = jnp.stack(
            [
                jnp.arange(expected).reshape(obs_shape),
                jnp.ones(obs_shape),
                jnp.zeros(obs_shape),
            ],
            axis=0,
        )

        def scan(carry, a):
            st, k = carry
            k, sk = jr.split(k)
            st, ob, *_ = w.step(st, a, key=sk)
            return (st, k), ob

        (_, _), obs_seq = lax.scan(scan, (state, scan_key), actions)
        assert obs_seq.shape == (actions.shape[0], expected)
        assert jnp.array_equal(obs_seq[0], actions[0].ravel())

    def test_jit(self):
        key = jr.key(0)
        base_env, state = eqx.nn.make_with_state(EchoEnv)(
            space=Box(-jnp.inf, jnp.inf, shape=(2, 2))
        )
        w = FlattenObservation(env=base_env)

        @eqx.filter_jit
        def f(ww, st, k):
            st, ob, _ = ww.reset(st, key=k)
            return ob

        ob = f(w, state, key)
        assert ob.shape == (4,)
