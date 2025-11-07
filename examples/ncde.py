import jax
from jax import random as jr

from lerax.algorithm import PPO
from lerax.env import CartPole
from lerax.policy import NCDEActorCriticPolicy
from lerax.wrapper import TimeLimit

jax.config.update("jax_enable_x64", True)

policy_key, learn_key = jr.split(jr.key(0), 2)

env = TimeLimit(CartPole(), max_episode_steps=512)
policy = NCDEActorCriticPolicy(env, key=policy_key)
algo = PPO()

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
    show_progress_bar=True,
    tb_log=True,
)
