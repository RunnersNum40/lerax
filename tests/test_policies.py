from __future__ import annotations

import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.policy import AbstractPolicy
from oryx.policy.actor_critic import AbstractActorCriticPolicy, CustomActorCriticPolicy
from oryx.space import Box

from .shared import DiscreteActionEnv, EchoEnv


class TestAbstractBases:
    def test_cannot_instantiate_abstract_policy(self):
        with pytest.raises(TypeError):
            AbstractPolicy()  # pyright: ignore

    def test_cannot_instantiate_abstract_actorcritic(self):
        with pytest.raises(TypeError):
            AbstractActorCriticPolicy()  # pyright: ignore


class TestCustomActorCriticPolicyBox:
    def test_init_and_spaces(self):
        env = EchoEnv(
            action_space=Box(-jnp.ones(3), jnp.ones(3)),
            observation_space=Box(-jnp.inf, jnp.inf, shape=(5,)),
        )
        key = jr.key(0)
        policy, _ = eqx.nn.make_with_state(CustomActorCriticPolicy)(env=env, key=key)

        assert policy.action_space == env.action_space
        assert policy.observation_space == env.observation_space
        assert policy.log_std.shape == env.action_space.shape

    def test_predict_bounds_and_shapes(self):
        init_key, sample_key = jr.split(jr.key(0), 2)
        env = EchoEnv(
            action_space=Box(-jnp.ones(2), jnp.ones(2)),
            observation_space=Box(-jnp.inf, jnp.inf, shape=(4,)),
        )
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=init_key
        )

        obs = jnp.arange(4.0)
        for key in jr.split(sample_key, 5):
            state, action = policy.predict(state, obs, key=key)
            assert action.shape == env.action_space.shape
            assert env.action_space.contains(action)

    def test_call_and_evaluate_action_consistency(self):
        key = jr.key(0)
        env = EchoEnv(
            action_space=Box(-2.0 * jnp.ones(3), 2.0 * jnp.ones(3)),
            observation_space=Box(-jnp.inf, jnp.inf, shape=(6,)),
        )
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=key
        )
        obs = jnp.linspace(-1.0, 1.0, 6)

        state, act, val, logp = policy(state, obs)
        assert act.shape == env.action_space.shape
        assert env.action_space.contains(act)
        assert val.shape == ()
        assert logp.shape == ()

        state, val2, logp2, ent = policy.evaluate_action(state, obs, act)
        assert val2.shape == ()
        assert logp2.shape == ()
        assert ent.shape == ()
        assert jnp.allclose(val2, val)
        assert jnp.allclose(logp2, logp)

    def test_reset_is_noop_for_default_models(self):
        key = jr.key(0)
        env = EchoEnv(
            action_space=Box(-jnp.ones(1), jnp.ones(1)),
            observation_space=Box(-jnp.inf, jnp.inf, shape=(2,)),
        )
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=key
        )
        state2 = policy.reset(state)
        assert state.tree_flatten() == state2.tree_flatten()


class TestCustomActorCriticPolicyDiscrete:
    def test_predict_and_evaluate(self):
        key = jr.key(0)
        env_key, pol_key, obs_key = jr.split(key, 3)
        env = DiscreteActionEnv(key=env_key, n_actions=4, obs_size=3)
        policy, state = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=pol_key
        )

        state, obs, _ = env.reset(state, key=obs_key)

        state, action = policy.predict(state, obs)
        assert isinstance(int(action), int)
        assert env.action_space.contains(action)

        keys = jr.split(jr.key(6), 8)
        actions = []
        for k in keys:
            state, a = policy.predict(state, obs, key=k)
            actions.append(int(a))
            assert env.action_space.contains(a)

        assert all(0 <= a < int(env.action_space.n) for a in actions)

        state, val, logp, ent = policy.evaluate_action(state, obs, (action))
        assert val.shape == ()
        assert logp.shape == ()
        assert ent.shape == ()
