from __future__ import annotations

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr

from oryx.algorithm import PPO
from oryx.policy.actor_critic import CustomActorCriticPolicy
from oryx.space import Box
from tests.envs import EchoEnv


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
