from __future__ import annotations

import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.env import AbstractEnv, AbstractEnvLike
from oryx.wrapper import Identity
from tests.envs import DummyEnv


class TestAbstractEnvLike:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractEnvLike()  # pyright: ignore

    def test_missing_methods(self):
        class NoStepEnv(AbstractEnvLike):
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


class TestAbstractEnv:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractEnv()  # pyright: ignore

    def test_missing_methods(self):
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

    def test_dummy_env_reset_and_step_shapes(self):
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

    def test_unwrapped_returns_base_env(self):
        base_env, _ = eqx.nn.make_with_state(DummyEnv)(key=jr.key(0))
        wrapped_env = Identity(base_env)

        assert wrapped_env.unwrapped is base_env
