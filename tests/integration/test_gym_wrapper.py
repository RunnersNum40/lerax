import gymnasium as gym
from jax import random as jr

from lerax.algorithm import PPO
from lerax.compatibility.gym import GymToLeraxEnv
from lerax.policy import MLPActorCriticPolicy


def test_gym_environment_training():
    """Test PPO training with a wrapped Gymnasium environment."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    gym_env = gym.make("CartPole-v1")
    env = GymToLeraxEnv(gym_env)
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = PPO(num_envs=1)  # Vectorization is not supported for Gym environments

    trained_policy = algo.learn(env, policy, total_timesteps=512, key=learn_key)

    assert trained_policy is not None
