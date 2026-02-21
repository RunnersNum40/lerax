import gymnasium as gym
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ConsoleBackend, LoggingCallback, TensorBoardBackend
from lerax.compatibility.gym import GymToLeraxEnv
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

gym_env = gym.make("CartPole-v1")
env = GymToLeraxEnv(gym_env)
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO(num_envs=1)  # Vectorization is not supported for Gym environments
logger = LoggingCallback(
    [TensorBoardBackend(), ConsoleBackend(total_timesteps=2**16)],
    env=env,
    policy=policy,
)

policy = algo.learn(env, policy, total_timesteps=2**16, key=learn_key, callback=logger)
logger.close()
