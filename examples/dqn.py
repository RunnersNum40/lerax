from jax import random as jr

from lerax.algorithm import DQN
from lerax.callback import ProgressBarCallback, TensorBoardCallback
from lerax.env import CartPole
from lerax.policy import MLPDQNPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPDQNPolicy(env=env, key=policy_key)
algo = DQN()
callbacks = [ProgressBarCallback(2**16), TensorBoardCallback(env=env, policy=policy)]

policy = algo.learn(
    env, policy, total_timesteps=2**14, key=learn_key, callbacks=callbacks
)
