from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import SAC
from lerax.benchmark import average_reward
from lerax.env.classic_control import Pendulum
from lerax.policy import MLPSACPolicy


def test_sac_trains():
    """Test SAC training completes without error."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = Pendulum()
    policy = MLPSACPolicy(env, feature_size=32, width_size=32, depth=1, key=policy_key)
    algo = SAC(
        buffer_size=512,
        learning_starts=128,
        num_steps=1,
        batch_size=32,
        policy_frequency=2,
        q_width_size=32,
        q_depth=1,
    )

    trained_policy = algo.learn(env, policy, total_timesteps=256, key=learn_key)

    assert trained_policy is not None


def test_sac_without_autotune():
    """Test SAC training with fixed alpha completes without error."""
    policy_key, learn_key = jr.split(jr.key(42), 2)

    env = Pendulum()
    policy = MLPSACPolicy(env, feature_size=32, width_size=32, depth=1, key=policy_key)
    algo = SAC(
        buffer_size=512,
        learning_starts=128,
        num_steps=1,
        batch_size=32,
        policy_frequency=2,
        autotune=False,
        initial_alpha=0.2,
        q_width_size=32,
        q_depth=1,
    )

    trained_policy = algo.learn(env, policy, total_timesteps=256, key=learn_key)

    assert trained_policy is not None


def test_sac_learns():
    """Test that SAC training improves policy performance."""
    policy_key, learn_key, eval_key = jr.split(jr.key(7), 3)

    env = Pendulum()
    untrained_policy = MLPSACPolicy(env=env, key=policy_key)

    untrained_reward = average_reward(
        env,
        untrained_policy,
        num_episodes=8,
        max_steps=200,
        deterministic=True,
        key=eval_key,
    )

    algo = SAC(
        num_envs=1,
        num_steps=1,
        buffer_size=8192,
        batch_size=64,
        learning_starts=512,
        q_width_size=64,
        q_depth=2,
    )

    trained_policy = algo.learn(
        env, untrained_policy, total_timesteps=16384, key=learn_key
    )

    trained_reward = average_reward(
        env,
        trained_policy,
        num_episodes=8,
        max_steps=200,
        deterministic=True,
        key=eval_key,
    )

    assert jnp.isfinite(untrained_reward)
    assert jnp.isfinite(trained_reward)
    assert trained_reward >= untrained_reward
