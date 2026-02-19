from jax import random as jr

from lerax.algorithm import SAC
from lerax.env import Pendulum
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
