from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.policy import AbstractPolicy
from oryx.policy.actor_critic import AbstractActorCriticPolicy, CustomActorCriticPolicy
from oryx.space import Box
from oryx.utils import clone_state

from .shared.envs import DiscreteActionEnv, EchoEnv
from .shared.models import Doubler, StatefulDoubler


class TestAbstractPolicy:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractPolicy()  # pyright: ignore


class TestAbstractActorCriticPolicy:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractActorCriticPolicy()  # pyright: ignore


class TestCustomActorCriticPolicyBox:
    def test_interface(self):
        env = EchoEnv(
            space=Box(-jnp.ones(3), jnp.ones(3)),
        )
        key = jr.key(0)
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=key
        )

        assert policy.action_space == env.action_space
        assert policy.observation_space == env.observation_space
        assert policy.log_std.shape == env.action_space.shape

        obs = jnp.linspace(-1.0, 1.0, env.observation_space.shape[0])

        state, action = policy.predict(state, obs, key=jr.key(1))
        assert action.shape == env.action_space.shape
        assert env.action_space.contains(action)

        state, act2, val, logp = policy(state, obs, key=jr.key(2))
        assert act2.shape == env.action_space.shape
        assert env.action_space.contains(act2)
        assert val.shape == ()
        assert logp.shape == ()

        state, v2, lp2, ent = policy.evaluate_action(state, obs, act2)
        assert v2.shape == ()
        assert lp2.shape == ()
        assert ent.shape == ()

        state2 = policy.reset(state)
        assert jax.tree.flatten(state) == jax.tree.flatten(state2)

    def test_jit(self):
        init_key, step_key = jr.split(jr.key(0))
        env = EchoEnv(
            space=Box(-2.0 * jnp.ones(2), 2.0 * jnp.ones(2)),
        )
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=init_key
        )
        obs = jnp.arange(2)

        @eqx.filter_jit
        def f(pol, st, ob, key):
            st, act, val, logp = pol(st, ob, key=key)
            st, v2, lp2, ent = pol.evaluate_action(st, ob, act)
            return st, act, val, logp, v2, lp2, ent

        state, act, val, logp, v2, lp2, ent = f(policy, state, obs, step_key)
        assert env.action_space.contains(act)
        assert val.shape == ()
        assert logp.shape == ()
        assert v2.shape == ()
        assert lp2.shape == ()
        assert ent.shape == ()


class TestCustomActorCriticPolicyDiscrete:
    def test_interface(self):
        key = jr.key(0)
        env_key, pol_key, obs_key = jr.split(key, 3)
        env = DiscreteActionEnv(key=env_key, n_actions=4, obs_size=3)
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=pol_key
        )

        state, obs, _ = env.reset(state, key=obs_key)

        state, action = policy.predict(state, obs, key=jr.key(1))
        assert isinstance(int(action), int)
        assert env.action_space.contains(action)

        state, val, logp, ent = policy.evaluate_action(state, obs, action)
        assert val.shape == ()
        assert logp.shape == ()
        assert ent.shape == ()

        state = policy.reset(state)

    def test_jit(self):
        key = jr.key(0)
        env_key, pol_key, obs_key, call_key = jr.split(key, 4)
        env = DiscreteActionEnv(key=env_key, n_actions=5, obs_size=3)
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=pol_key
        )
        state, obs, _ = env.reset(state, key=obs_key)

        @eqx.filter_jit
        def step(pol, st, ob, key):
            st, act, val, logp = pol(st, ob, key=key)
            st, v2, lp2, ent = pol.evaluate_action(st, ob, act)
            return st, act, val, logp, v2, lp2, ent

        state, action, val, logp, v2, lp2, ent = step(policy, state, obs, call_key)
        assert env.action_space.contains(action)
        assert val.shape == ()
        assert logp.shape == ()
        assert v2.shape == ()
        assert lp2.shape == ()
        assert ent.shape == ()


class TestCustomActorCriticPolicyPrivate:
    def test_apply_model_stateless(self):
        model = Doubler()
        state = cast(eqx.nn.State, None)

        _, y = CustomActorCriticPolicy._apply_model(state, model, jnp.array(2.0))
        assert jnp.allclose(y, 4.0)

    def test_apply_model_stateful(self):
        model, state = eqx.nn.make_with_state(StatefulDoubler)()
        s1 = clone_state(state)
        s2, y = CustomActorCriticPolicy._apply_model(state, model, jnp.array(5.0))
        assert jnp.allclose(y, 10.0)
        assert not (jax.tree.flatten(s1) == jax.tree.flatten(s2))

    def test_reset_stateful_model(self):
        env_key, policy_key = jr.split(jr.key(0), 2)
        model = StatefulDoubler()
        env = EchoEnv()
        with pytest.raises(ValueError):
            eqx.nn.make_with_state(CustomActorCriticPolicy)(
                env=env, feature_extractor=model, key=policy_key
            )

        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, value_model=model, key=policy_key
        )

        state, obs, _ = env.reset(state, key=env_key)
        state = policy.reset(state)
