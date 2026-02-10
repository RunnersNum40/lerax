from jax import random as jr

from lerax.algorithm import REINFORCE
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy


def test_reinforce_trains():
    """Test REINFORCE training completes without error."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = REINFORCE(num_steps=64)

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_reinforce_multi_env():
    """Test REINFORCE with multiple parallel environments."""
    policy_key, learn_key = jr.split(jr.key(1), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = REINFORCE(num_envs=2, num_steps=32)

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None
