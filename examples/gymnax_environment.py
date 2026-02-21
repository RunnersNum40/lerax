import gymnax as gym
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ConsoleBackend, LoggingCallback, TensorBoardBackend
from lerax.compatibility.gymnax import GymnaxToLeraxEnv
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

gymnax_env, params = gym.make("CartPole-v1")
env = GymnaxToLeraxEnv(gymnax_env, params)

policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()
logger = LoggingCallback(
    [TensorBoardBackend(), ConsoleBackend(total_timesteps=2**16)],
    env=env,
    policy=policy,
)

policy = algo.learn(env, policy, total_timesteps=2**16, key=learn_key, callback=logger)
logger.close()
