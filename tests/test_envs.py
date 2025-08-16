from __future__ import annotations

import equinox as eqx
import pytest
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from oryx.env import AbstractEnv, AbstractEnvLike, CartPole
from oryx.space import Box, Discrete

from .shared import DummyEnv


def test_cannot_instantiate_abstract_envs():
    """Direct instantiation of abstract bases must fail."""
    with pytest.raises(TypeError):
        AbstractEnvLike()  # pyright: ignore
    with pytest.raises(TypeError):
        AbstractEnv()  # pyright: ignore


def test_missing_methods_detected():
    """A subclass that omits a required method should be abstract."""

    class NoStepEnv(AbstractEnv):
        def reset(self, state, *, key):
            raise NotImplementedError

        @property
        def action_space(self):
            raise NotImplementedError

        @property
        def observation_space(self):
            raise NotImplementedError

        def render(self, state): ...
        def close(self): ...

    with pytest.raises(TypeError):
        NoStepEnv()  # pyright: ignore


def test_dummy_env_reset_and_step_shapes():
    """`reset` and `step` should respect the declared spaces & signatures."""
    key = jr.key(0)
    env, state = eqx.nn.make_with_state(DummyEnv)(key=key)

    key, reset_key = jr.split(key)
    state, obs, info = env.reset(state, key=reset_key)
    assert env.observation_space.contains(obs)
    assert info == {}

    action = jnp.asarray(0.5)
    key, step_key = jr.split(key)
    out = env.step(state, action, key=step_key)
    assert len(out) == 6
    _state, new_obs, reward, term, trunc, info = out
    assert env.observation_space.contains(new_obs)
    assert isinstance(reward, jnp.ndarray) and reward.shape == ()
    assert isinstance(term, jnp.ndarray) and term.shape == ()
    assert isinstance(trunc, jnp.ndarray) and trunc.shape == ()
    assert info == {}


def test_unwrapped_returns_base_env():
    """Even after extra wrapping, `.unwrapped` should reach the original env."""
    from oryx.wrapper.base_wrapper import AbstractNoRenderOrCloseWrapper

    class IdentityWrapper(AbstractNoRenderOrCloseWrapper):
        env: DummyEnv

        def __init__(self, env):
            self.env = env

        def reset(self, state, *, key):
            return self.env.reset(state, key=key)

        def step(self, state, action, *, key):
            return self.env.step(state, action, key=key)

        @property
        def action_space(self):
            return self.env.action_space

        @property
        def observation_space(self):
            return self.env.observation_space

    key = jr.key(0)
    base_env, _ = eqx.nn.make_with_state(DummyEnv)(key=key)
    wrapped_env = IdentityWrapper(base_env)

    assert wrapped_env.unwrapped is base_env, "unwrapped must expose base env"


def test_cartpole():
    key, reset_key = jr.split(jr.key(0), 2)
    env, state = eqx.nn.make_with_state(CartPole)()

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


def test_cartpole_scan():
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
