from __future__ import annotations

import equinox as eqx
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.env import MountainCar
from oryx.space import Box, Discrete


class TestMountainCar:
    def test_mountaincar(self):
        key, reset_key = jr.split(jr.key(0), 2)
        env, state = eqx.nn.make_with_state(MountainCar)()

        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (2,)

        state, obs, info = env.reset(state, key=reset_key)
        assert obs.shape == (2,)
        assert info == {}
        assert env.observation_space.contains(obs)

        key, step_key = jr.split(key)
        action = jnp.asarray(1)
        state, obs2, reward, terminated, truncated, info2 = env.step(
            state, action, key=step_key
        )
        assert obs2.shape == (2,)
        assert reward.shape == ()
        assert terminated.shape == ()
        assert truncated.shape == ()
        assert pytest.approx(float(reward)) == -1.0
        assert isinstance(info2, dict)

        state = state.set(env.state_index, jnp.asarray([env.min_position, -0.01]))
        key, step_key = jr.split(key)
        state, obs3, *_ = env.step(state, jnp.asarray(1), key=step_key)
        assert obs3.shape == (2,)
        assert env.min_position <= float(obs3[0]) <= env.max_position
        assert abs(float(obs3[1])) <= env.max_speed + 1e-6

    def test_mountaincar_scan(self):
        key, reset_key = jr.split(jr.key(0), 2)
        env, state = eqx.nn.make_with_state(MountainCar)()
        state, obs0, _ = env.reset(state, key=reset_key)

        def scan_step(carry, _):
            st, obs, k = carry
            k, step_k = jr.split(k)
            act = jnp.asarray(1)
            st, obs, rew, term, trunc, _ = env.step(st, act, key=step_k)
            return (st, obs, k), (obs, rew, term, trunc)

        (state, _, _), (obs_seq, rew_seq, term_seq, trunc_seq) = lax.scan(
            scan_step, (state, obs0, key), None, length=8
        )
        assert obs_seq.shape == (8, 2)
        assert rew_seq.shape == (8,)
        assert term_seq.shape == (8,)
        assert trunc_seq.shape == (8,)
