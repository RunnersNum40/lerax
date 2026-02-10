from jax import random as jr

from lerax.algorithm import A2C
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy


def test_a2c_trains():
    """Test A2C training completes without error."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = A2C(num_envs=1, num_steps=64)

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_a2c_multi_env():
    """Test A2C with multiple parallel environments."""
    policy_key, learn_key = jr.split(jr.key(1), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = A2C(num_envs=2, num_steps=32)

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_a2c_with_gae():
    """Test A2C with configurable GAE lambda."""
    policy_key, learn_key = jr.split(jr.key(2), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = A2C(num_envs=1, num_steps=64, gae_lambda=0.95)

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None
