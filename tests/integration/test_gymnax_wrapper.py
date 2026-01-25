import gymnax
from jax import random as jr

from lerax.algorithm import PPO
from lerax.compatibility.gymnax import GymnaxToLeraxEnv
from lerax.policy import MLPActorCriticPolicy


def test_gymnax_environment_training():
    """Test PPO training with a wrapped Gymnax environment."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    gymnax_env, params = gymnax.make("CartPole-v1")
    env = GymnaxToLeraxEnv(gymnax_env, params)

    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = PPO()

    trained_policy = algo.learn(env, policy, total_timesteps=512, key=learn_key)

    assert trained_policy is not None
