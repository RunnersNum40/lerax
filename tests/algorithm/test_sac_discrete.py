from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import SACDiscrete
from lerax.benchmark import average_reward
from lerax.env.classic_control import CartPole
from lerax.policy.sac.discrete import MLPDiscreteSACPolicy


def test_sac_discrete_trains():
    """Test SAC-Discrete training completes without error."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = CartPole()
    policy = MLPDiscreteSACPolicy(env, width_size=32, depth=1, key=policy_key)
    algo = SACDiscrete(
        buffer_size=256,
        learning_starts=64,
        num_steps=4,
        batch_size=32,
        q_width_size=32,
        q_depth=1,
    )

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_sac_discrete_without_autotune():
    """Test SAC-Discrete training with fixed alpha completes without error."""
    policy_key, learn_key = jr.split(jr.key(42), 2)

    env = CartPole()
    policy = MLPDiscreteSACPolicy(env, width_size=32, depth=1, key=policy_key)
    algo = SACDiscrete(
        buffer_size=256,
        learning_starts=64,
        num_steps=4,
        batch_size=32,
        autotune=False,
        initial_alpha=0.2,
        q_width_size=32,
        q_depth=1,
    )

    trained_policy = algo.learn(env, policy, total_timesteps=128, key=learn_key)

    assert trained_policy is not None


def test_sac_discrete_learns():
    """Test that SAC-Discrete training improves policy performance."""
    policy_key, learn_key, eval_key = jr.split(jr.key(7), 3)

    env = CartPole()
    untrained_policy = MLPDiscreteSACPolicy(env=env, key=policy_key)

    untrained_reward = average_reward(
        env,
        untrained_policy,
        num_episodes=8,
        max_steps=500,
        deterministic=True,
        key=eval_key,
    )

    algo = SACDiscrete(
        num_envs=1,
        num_steps=1,
        buffer_size=4096,
        batch_size=32,
        learning_starts=512,
        q_width_size=64,
        q_depth=2,
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
