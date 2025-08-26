from __future__ import annotations

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.algorithm import PPO
from oryx.policy.actor_critic import CustomActorCriticPolicy
from oryx.space import Box
from tests.buffers import empty_buffer
from tests.envs import EchoEnv
from tests.policies import ConstantPolicy


def _make_ppo(num_steps: int = 8) -> tuple[PPO, eqx.nn.State]:
    key = jr.key(0)
    env = EchoEnv(space=Box(-jnp.ones(2), jnp.ones(2)))
    policy, _ = eqx.nn.make_with_state(CustomActorCriticPolicy)(env=env, key=key)
    algo, state = eqx.nn.make_with_state(PPO)(
        env=env,
        policy=policy,
        num_steps=num_steps,
        num_epochs=2,
        num_mini_batches=2,
    )
    return algo, state


class TestPPOLossAndTraining:
    def test_train_batch(self):
        algo, state = _make_ppo(num_steps=8)
        k_init, k_roll = jr.split(jr.key(1))
        state, carry = algo.initialize_iteration_carry(state, key=k_init)
        state, step_carry, buf, _ = algo.collect_rollout(
            algo.policy, state, carry.step_carry, key=k_roll
        )

        batched = buf.batches(algo.batch_size, key=None)
        single = jax.tree.map(lambda x: x[0], batched)

        state, policy, stats = algo.train_batch(state, algo.policy, single)
        for s in (
            stats.approx_kl,
            stats.total_loss,
            stats.policy_loss,
            stats.value_loss,
        ):
            assert s.shape == ()
            assert jnp.isfinite(s)

        lr = algo.learning_rate(state)
        assert lr.shape == ()

    def test_train_epoch_and_train(self):
        algo, state = _make_ppo(num_steps=8)
        k_init, k_roll, k_epoch, k_train = jr.split(jr.key(2), 4)
        state, carry = algo.initialize_iteration_carry(state, key=k_init)
        state, _, buf, _ = algo.collect_rollout(
            algo.policy, state, carry.step_carry, key=k_roll
        )

        state, pol_after, epoch_stats = algo.train_epoch(
            state, algo.policy, buf, key=k_epoch
        )
        assert jnp.isfinite(epoch_stats.total_loss)

        state, pol_after2, log = algo.train(state, pol_after, buf, key=k_train)
        required = {
            "loss/approx_kl",
            "loss/total",
            "loss/policy",
            "loss/value",
            "loss/entropy",
            "loss/state_magnitude",
            "stats/variance",
            "stats/explained_variance",
        }
        assert required.issubset(log.keys())
        assert all(jnp.isfinite(v) for v in log.values())

    def test_learn_integration(self):
        algo, state = _make_ppo(num_steps=8)
        state, policy = algo.learn(state, total_timesteps=8, key=jr.key(3))
        env = algo.env
        state, obs, _ = env.reset(state.substate(env), key=jr.key(4))
        state = state.update(state.substate(algo.policy))
        pol_state = state.substate(policy)
        pol_state, act = policy.predict(pol_state, obs, key=jr.key(5))
        assert env.action_space.contains(act)

    def test_ppo_loss_clip_vs_unclipped_and_entropy(self):
        buf = empty_buffer(n=6)
        pol = ConstantPolicy(new_value=5.0, logp=0.0, entropy_val=2.0)

        loss_c, stats_c = PPO.ppo_loss(
            pol,
            buf,
            normalize_advantages=False,
            clip_coefficient=0.1,
            clip_value_loss=True,
            value_loss_coefficient=0.5,
            state_magnitude_coefficient=0.0,
            entropy_loss_coefficient=0.01,
        )
        policy_loss = -1.0
        value_loss_clipped = 0.5 * (0.1**2)
        entropy = 2.0
        expected_total = policy_loss + 0.5 * value_loss_clipped - 0.01 * entropy

        assert float(stats_c.policy_loss) == pytest.approx(policy_loss)
        assert float(stats_c.value_loss) == pytest.approx(value_loss_clipped)
        assert float(stats_c.entropy_loss) == pytest.approx(entropy)
        assert float(stats_c.state_magnitude_loss) == pytest.approx(0.0)
        assert float(stats_c.total_loss) == pytest.approx(expected_total, rel=1e-6)
        assert float(stats_c.approx_kl) == pytest.approx(0.0)

        loss_u, stats_u = PPO.ppo_loss(
            pol,
            buf,
            normalize_advantages=False,
            clip_coefficient=0.1,
            clip_value_loss=False,
            value_loss_coefficient=0.5,
            state_magnitude_coefficient=0.0,
            entropy_loss_coefficient=0.01,
        )
        assert float(stats_u.value_loss) == pytest.approx(12.5)
        expected_total_unclipped = policy_loss + 0.5 * 12.5 - 0.01 * entropy
        assert float(stats_u.total_loss) == pytest.approx(
            expected_total_unclipped, rel=1e-6
        )
        assert expected_total_unclipped != pytest.approx(expected_total)


def _setup_algo(*, anneal: bool):
    key = jr.key(0)
    env = EchoEnv(space=Box(-jnp.ones(2), jnp.ones(2)))
    policy, _ = eqx.nn.make_with_state(CustomActorCriticPolicy)(env=env, key=key)
    algo, state = eqx.nn.make_with_state(PPO)(
        env=env,
        policy=policy,
        num_steps=8,
        num_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        anneal_learning_rate=anneal,
    )
    return algo, state


def _collect_once(algo, state, *, key):
    state, carry = algo.initialize_iteration_carry(state, key=key)
    state, _, buf, _ = algo.collect_rollout(
        algo.policy, state, carry.step_carry, key=jr.split(key)[0]
    )
    return state, buf


def test_learning_rate_anneals_down():
    algo, state = _setup_algo(anneal=True)
    state, buf = _collect_once(algo, state, key=jr.key(1))

    lr0 = float(algo.learning_rate(state))
    state, _, _ = algo.train(state, algo.policy, buf, key=jr.key(2))
    lr1 = float(algo.learning_rate(state))
    state, _, _ = algo.train(state, algo.policy, buf, key=jr.key(3))
    lr2 = float(algo.learning_rate(state))

    assert lr0 > lr1 > lr2


def test_learning_rate_constant_when_no_anneal():
    algo, state = _setup_algo(anneal=False)
    state, buf = _collect_once(algo, state, key=jr.key(4))

    lr0 = float(algo.learning_rate(state))
    state, _, _ = algo.train(state, algo.policy, buf, key=jr.key(5))
    lr1 = float(algo.learning_rate(state))

    assert lr0 == pytest.approx(1e-3)
    assert lr1 == pytest.approx(lr0)
