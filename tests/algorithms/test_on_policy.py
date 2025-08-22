from __future__ import annotations

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Float

from oryx.algorithm import AbstractOnPolicyAlgorithm
from oryx.env import AbstractEnvLike
from oryx.policy.actor_critic import AbstractActorCriticPolicy, CustomActorCriticPolicy
from oryx.space import Box
from tests.envs import DiscreteActionEnv, EchoEnv


class OnPolicyAlgorithm[ActType, ObsType](AbstractOnPolicyAlgorithm[ActType, ObsType]):
    """Minimal concrete on-policy algorithm just for testing."""

    env: AbstractEnvLike[ActType, ObsType]
    policy: AbstractActorCriticPolicy[Float, ActType, ObsType]
    state_index: eqx.nn.StateIndex[None] = eqx.nn.StateIndex(None)

    gae_lambda: float = 0.0
    gamma: float = 0.0
    num_steps: int = 8
    batch_size: int = 2

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
    ):
        self.env = env
        self.policy = policy

    def train(self, state, policy, rollout_buffer, *, key):
        log = {
            "loss": jnp.asarray(0.0),
        }
        return state, policy, log

    def learning_rate(self, state):
        return jnp.asarray(0.0)

    def save(self, path: str): ...

    @classmethod
    def load(cls, path):
        raise NotImplementedError


class TestOnPolicyAlgorithm:
    def test_env_policy_init(self):
        key_env, key_pol = jr.split(jr.key(0), 2)
        env = DiscreteActionEnv(key=key_env, n_actions=3, obs_size=3)

        box_env = EchoEnv(space=Box(-jnp.ones(()), jnp.ones(())))
        policy, _ = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=box_env, key=key_pol
        )

        with pytest.raises(ValueError):
            eqx.nn.make_with_state(OnPolicyAlgorithm)(env=env, policy=policy)

    def test_collect_rollout_shapes_and_finiteness(self):
        policy_key, init_key, rollout_key = jr.split(jr.key(0), 3)
        env = EchoEnv(space=Box(-jnp.ones(2), jnp.ones(2)))
        policy, _ = eqx.nn.make_with_state(CustomActorCriticPolicy)(
            env=env, key=policy_key
        )
        algo, state = eqx.nn.make_with_state(OnPolicyAlgorithm)(env=env, policy=policy)

        state, carry = algo.initialize_iteration_carry(state, key=init_key)
        state, step_carry, buf, ep = algo.collect_rollout(
            algo.policy, state, carry.step_carry, key=rollout_key
        )

        assert isinstance(algo, AbstractOnPolicyAlgorithm)
        assert buf.shape == (algo.num_steps,)
        assert not jnp.isnan(buf.values).any()
        assert not jnp.isnan(buf.log_probs).any()
        assert not jnp.isnan(buf.returns).any()
        assert not jnp.isnan(buf.advantages).any()
        assert ep is None

        first_obs = jax.tree.map(lambda x: x[0], buf.observations)
        first_act = jax.tree.map(lambda x: x[0], buf.actions)
        assert first_obs.shape == algo.env.observation_space.shape
        assert first_act.shape == algo.env.action_space.shape
        assert step_carry.step_count == algo.num_steps
