from jax import random as jr

from lerax.algorithm import DQN
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
