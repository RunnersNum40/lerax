from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import A2C
from lerax.benchmark import average_reward
from lerax.env.classic_control import CartPole
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


def test_a2c_learns():
    """Test that A2C training improves policy performance."""
    policy_key, learn_key, eval_key = jr.split(jr.key(7), 3)

    env = CartPole()
    untrained_policy = MLPActorCriticPolicy(env=env, key=policy_key)

    untrained_reward = average_reward(
        env,
        untrained_policy,
        num_episodes=8,
        max_steps=500,
        deterministic=True,
        key=eval_key,
    )

    algo = A2C(num_envs=4, num_steps=8)

    trained_policy = algo.learn(
        env, untrained_policy, total_timesteps=4096, key=learn_key
    )

    trained_reward = average_reward(
        env,
        trained_policy,
        num_episodes=8,
        max_steps=500,
        deterministic=True,
        key=eval_key,
    )

    assert jnp.isfinite(untrained_reward)
    assert jnp.isfinite(trained_reward)
    assert trained_reward >= untrained_reward
