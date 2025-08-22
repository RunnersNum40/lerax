from __future__ import annotations

import equinox as eqx
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.env import ContinuousMountainCar
from oryx.space import Box


class TestContinuousMountainCar:
    def test_continuous_mountaincar(self):
        key, reset_key = jr.split(jr.key(0), 2)
        env, state = eqx.nn.make_with_state(ContinuousMountainCar)()

        assert isinstance(env.action_space, Box)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (2,)

        state, obs, info = env.reset(state, key=reset_key)
        assert obs.shape == (2,)
        assert info == {}
        assert env.observation_space.contains(obs)

        state = state.set(env.state_index, jnp.asarray([env.min_position, 0.0]))
        key, step_key = jr.split(key)
        state, obs2, reward2, terminated2, truncated2, _ = env.step(
            state, jnp.asarray(0.0), key=step_key
        )
        assert obs2.shape == (2,)
        assert not bool(terminated2)
        assert pytest.approx(float(reward2)) == 100.0
        assert truncated2.shape == ()

        state = state.set(
            env.state_index, jnp.asarray([env.max_position, env.goal_velocity])
        )
        key, step_key = jr.split(key)
        state, obs3, reward3, terminated3, truncated3, _ = env.step(
            state, jnp.asarray(0.0), key=step_key
        )
        assert obs3.shape == (2,)
        assert bool(terminated3)
        assert pytest.approx(float(reward3)) == 0.0
        assert truncated3.shape == ()

    def test_continuous_mountaincar_scan(self):
        key, reset_key = jr.split(jr.key(0), 2)
        env, state = eqx.nn.make_with_state(ContinuousMountainCar)()
        state, obs0, _ = env.reset(state, key=reset_key)

        def scan_step(carry, _):
            st, obs, k = carry
            k, step_k = jr.split(k)
            act = jnp.asarray(0.0)
            st, obs, rew, term, trunc, _ = env.step(st, act, key=step_k)
            return (st, obs, k), (obs, rew, term, trunc)

        (state, _, _), (obs_seq, rew_seq, term_seq, trunc_seq) = lax.scan(
            scan_step, (state, obs0, key), None, length=8
        )
        assert obs_seq.shape == (8, 2)
        assert rew_seq.shape == (8,)
        assert term_seq.shape == (8,)
        assert trunc_seq.shape == (8,)
