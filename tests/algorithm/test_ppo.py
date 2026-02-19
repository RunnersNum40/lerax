from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import PPO
from lerax.benchmark import average_reward
from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy


def test_ppo_trains():
    """Test PPO training completes without error."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = PPO(num_envs=1, num_steps=64)

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_ppo_multi_env():
    """Test PPO with multiple parallel environments."""
    policy_key, learn_key = jr.split(jr.key(1), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = PPO(num_envs=2, num_steps=32)

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_ppo_learns():
    """Test that PPO training improves policy performance."""
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

    algo = PPO(num_envs=4, num_steps=128, num_epochs=4, num_batches=8)

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
