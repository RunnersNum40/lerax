from __future__ import annotations

import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.env.base_env import AbstractEnv, AbstractEnvLike

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
    from oryx.wrappers.base_wrapper import AbstractNoRenderOrCloseWrapper

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

    key = jr.key(1)
    base_env, _ = eqx.nn.make_with_state(DummyEnv)(key=key)
    wrapped_env = IdentityWrapper(base_env)

    assert wrapped_env.unwrapped is base_env, "unwrapped must expose base env"
