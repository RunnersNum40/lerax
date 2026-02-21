from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ConsoleBackend, LoggingCallback, TensorBoardBackend
from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()
total_timesteps = 2**16
logger = LoggingCallback(
    [TensorBoardBackend(), ConsoleBackend(total_timesteps=total_timesteps)],
    env=env,
    policy=policy,
    video_interval=1,
)

policy = algo.learn(
    env, policy, total_timesteps=total_timesteps, key=learn_key, callback=logger
)
logger.close()
