from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import LoggingCallback, ProgressBarCallback, TensorBoardBackend
from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()
callbacks = [
    ProgressBarCallback(2**16),
    LoggingCallback(TensorBoardBackend(), video_interval=1),
]

policy = algo.learn(
    env, policy, total_timesteps=2**16, key=learn_key, callback=callbacks
)
