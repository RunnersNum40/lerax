from __future__ import annotations

import equinox as eqx
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.env import CartPole
from oryx.env.cartpole import SOLVER
from oryx.space import Box, Discrete


class TestCartPole:
    @pytest.mark.parametrize("solver", [SOLVER.explicit, SOLVER.implicit])
    def test_cartpole(self, solver):
        key, reset_key = jr.split(jr.key(0), 2)
        env, state = eqx.nn.make_with_state(CartPole)(solver=solver)

        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (4,)

        state, obs, info = env.reset(state, key=reset_key)
        assert obs.shape == (4,)
        assert info == {}

        key, step_key = jr.split(key)
        action = jnp.asarray(1)
        state, obs2, reward, terminated, truncated, info2 = env.step(
            state, action, key=step_key
        )

        assert obs2.shape == (4,)
        assert reward.shape == ()
        assert terminated.shape == ()
        assert truncated.shape == ()
        assert reward == 1.0
        assert isinstance(info2, dict)

        bad = jnp.array(
            [env.x_threshold * 3.0, 0.0, env.theta_threshold_radians * 3.0, 0.0]
        )
        state = state.set(env.state_index, bad)
        key, step_key = jr.split(key)
        state, *_ = env.step(state, action, key=step_key)

    def test_scan(self):
        key, reset_key = jr.split(jr.key(0), 2)
        env, state = eqx.nn.make_with_state(CartPole)()

        state, obs0, _ = env.reset(state, key=reset_key)

        def scan_step(carry, _):
            st, obs, k = carry
            k, step_k = jr.split(k)
            act = jnp.asarray(0)
            st, obs, rew, term, trunc, _ = env.step(st, act, key=step_k)
            return (st, obs, k), (obs, rew, term, trunc)

        (state, _, _), (obs_seq, rew_seq, term_seq, trunc_seq) = lax.scan(
            scan_step, (state, obs0, key), None, length=8
        )

        assert obs_seq.shape == (8, 4)
        assert rew_seq.shape == (8,)
        assert term_seq.shape == (8,)
        assert trunc_seq.shape == (8,)
