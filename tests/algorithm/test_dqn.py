from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import DQN
from lerax.benchmark import average_reward
from lerax.env import CartPole
from lerax.policy import MLPQPolicy


def test_dqn_trains():
    """Test DQN training completes without error."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = CartPole()
    policy = MLPQPolicy(env=env, key=policy_key)
    algo = DQN(
        buffer_size=256,
        learning_starts=64,
        num_steps=4,
        batch_size=32,
        target_update_interval=10,
    )

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_dqn_learns():
    """Test that DQN training improves policy performance."""
    policy_key, learn_key, eval_key = jr.split(jr.key(7), 3)

    env = CartPole()
    untrained_policy = MLPQPolicy(env=env, key=policy_key)

    untrained_reward = average_reward(
        env,
        untrained_policy,
        num_episodes=8,
        max_steps=500,
        deterministic=True,
        key=eval_key,
    )

    algo = DQN(
        num_envs=1,
        num_steps=1,
        buffer_size=4096,
        batch_size=32,
        learning_starts=512,
        target_update_interval=100,
    )

    trained_policy = algo.learn(
        env, untrained_policy, total_timesteps=8192, key=learn_key
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
