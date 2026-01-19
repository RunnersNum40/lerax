from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import PPO
from lerax.benchmark import average_reward
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy


def test_untrained_vs_trained_policy():
    """Test that training improves policy performance."""
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
