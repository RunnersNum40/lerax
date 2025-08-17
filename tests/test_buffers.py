from __future__ import annotations

import equinox as eqx
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.buffer import RolloutBuffer
from oryx.distribution import Normal
from oryx.policy import AbstractActorCriticPolicy
from oryx.utils import filter_scan

from .shared import EchoEnv


class TestRolloutBufferBasics:
    @staticmethod
    def _make_dummy_buffer(num_steps: int = 8):
        key = jr.key(0)
        observations = jnp.arange(num_steps) * 10.0
        actions = jnp.arange(num_steps) * -1.0
        rewards = jnp.arange(num_steps)
        terminations = jnp.full(num_steps, False)
        truncations = jnp.zeros_like(terminations)
        log_probs = jnp.zeros_like(rewards) - 0.5
        values = jnp.linspace(0.0, 1.0, num_steps)
        states = jnp.zeros(num_steps)
        return (
            RolloutBuffer(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                log_probs=log_probs,
                values=values,
                states=states,
            ),
            key,
        )

    def test_shape_property(self):
        buf, _ = self._make_dummy_buffer(11)
        assert buf.shape == (11,)

    def test_returns_and_advantages_are_populated_and_consistent(self):
        rewards = jnp.asarray([1.0, 2.0])
        values = jnp.asarray([0.5, 0.6])
        terminations = jnp.asarray([False, False])
        truncations = jnp.asarray([False, False])

        buf = RolloutBuffer(
            observations=jnp.zeros_like(rewards),
            actions=jnp.zeros_like(rewards),
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            log_probs=jnp.zeros_like(rewards),
            values=values,
            states=jnp.zeros_like(rewards),
        )

        last_value = jnp.asarray(0.7)
        done = jnp.asarray(False)

        gamma, lam = 0.99, 0.95
        buf = buf.compute_returns_and_advantages(
            last_value, done, gae_lambda=lam, gamma=gamma
        )

        assert not jnp.isnan(buf.advantages).any()
        assert not jnp.isnan(buf.returns).any()
        assert buf.advantages.shape == rewards.shape
        assert buf.returns.shape == rewards.shape
        assert jnp.allclose(buf.returns, buf.values + buf.advantages)

        expected_adv = jnp.asarray(
            [
                1.094 + 0.99 * 0.95 * 2.093,
                2.093,
            ]
        )
        expected_ret = expected_adv + values
        assert jnp.allclose(buf.advantages, expected_adv)
        assert jnp.allclose(buf.returns, expected_ret)

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_batches_shape_and_permutation(self, shuffle: bool):
        n_steps, batch_sz = 12, 4
        buf, key = self._make_dummy_buffer(n_steps)

        batched = buf.batches(batch_sz, key=key if shuffle else None)
        assert batched.rewards.shape == (n_steps // batch_sz, batch_sz)

        flat_after = batched.rewards.ravel()
        assert jnp.array_equal(jnp.sort(flat_after), buf.rewards)

        if shuffle:
            assert not jnp.array_equal(flat_after, buf.rewards)
        else:
            assert jnp.array_equal(flat_after, buf.rewards)

    def test_batches_handles_non_divisible_batch_size(self):
        buf, key = self._make_dummy_buffer(10)
        batch_sz = 3
        batched = buf.batches(batch_sz, key=key)
        expected_batches = 10 // batch_sz
        assert batched.rewards.shape[0] == expected_batches


class TestRolloutBufferWithEnv:

    class _SimplePolicy(AbstractActorCriticPolicy):
        state_index: eqx.nn.StateIndex[None]
        env: EchoEnv

        def __init__(self, env: EchoEnv):
            self.env = env
            self.state_index = eqx.nn.StateIndex(None)

        @property
        def action_space(self):
            return self.env.action_space

        @property
        def observation_space(self):
            return self.env.observation_space

        def extract_features(self, state, observation):
            return state, jnp.asarray(observation)

        def action_dist_from_features(self, state, features):
            return state, Normal(features, jnp.asarray(1.0))

        def value_from_features(self, state, features):
            return state, jnp.asarray(0.0)

        def reset(self, state):
            return state

    def test_env_policy_collection_and_batching(
        self, num_steps: int = 8, batch_size: int = 2
    ):
        reset_key, scan_key, batch_key = jr.split(jr.key(0), 3)

        env = EchoEnv()
        policy, state = eqx.nn.make_with_state(TestRolloutBufferWithEnv._SimplePolicy)(
            env=env
        )

        state, obs, _ = env.reset(state, key=reset_key)

        def step(carry, _):
            state, prev_obs, prev_term, prev_trunc, key = carry
            act_key, step_key, carry_key = jr.split(key, 3)

            policy_state = state.substate(policy)
            policy_state, action, value, log_prob = policy(
                policy_state, prev_obs, key=act_key
            )
            state = state.update(policy_state)

            state, obs, reward, termination, truncation, _ = env.step(
                state, action, key=step_key
            )

            out = RolloutBuffer(
                observations=prev_obs,
                actions=action,
                rewards=reward,
                terminations=prev_term,
                truncations=prev_trunc,
                log_probs=log_prob,
                values=value,
                states=state.substate(policy),
            )

            carry = (state, obs, termination, truncation, carry_key)
            return carry, out

        (_, obs, termination, truncation, _), rollout_buffer = filter_scan(
            step,
            (state, obs, jnp.asarray(False), jnp.asarray(False), scan_key),
            None,
            length=num_steps,
        )

        assert rollout_buffer.shape == (num_steps,)
        assert not jnp.isnan(rollout_buffer.values).any()
        assert not jnp.isnan(rollout_buffer.log_probs).any()

        _, last_value = policy.value(state, obs)
        rollout_buffer = rollout_buffer.compute_returns_and_advantages(
            last_value,
            jnp.logical_or(termination, truncation),
            gae_lambda=0.95,
            gamma=0.99,
        )
        assert not jnp.isnan(rollout_buffer.returns).any()
        assert not jnp.isnan(rollout_buffer.advantages).any()
        assert jnp.allclose(
            rollout_buffer.returns, rollout_buffer.values + rollout_buffer.advantages
        )

        batched = rollout_buffer.batches(batch_size, key=None)
        flat_obs = batched.observations.reshape(-1)
        flat_act = batched.actions.reshape(-1)
        assert jnp.allclose(flat_obs[1:], flat_act[:-1])
        assert batched.rewards.shape == (num_steps // batch_size, batch_size)

        batched = rollout_buffer.batches(batch_size, key=batch_key)
        flat_after = batched.actions.ravel()
        flat_before = rollout_buffer.actions[: (num_steps // batch_size) * batch_size]
        assert not jnp.array_equal(flat_after, flat_before)
        assert jnp.array_equal(jnp.sort(flat_after), jnp.sort(flat_before))
