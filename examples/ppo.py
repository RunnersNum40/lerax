from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ProgressBarCallback, TensorBoardCallback
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO(
    env=env,
    total_timesteps=2**16,
    callback=[ProgressBarCallback(2**16), TensorBoardCallback(env=env, policy=policy)],
)


policy = algo.learn(policy, key=learn_key)
