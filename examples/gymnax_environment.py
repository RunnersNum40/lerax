import gymnax as gym
from jax import random as jr

from lerax.algorithm import PPO
from lerax.compatibility.gymnax import GymnaxToLeraxEnv
from lerax.policy import MLPActorCriticPolicy
from lerax.wrapper import EpisodeStatistics, TimeLimit

policy_key, learn_key = jr.split(jr.key(0), 2)

gymnax_env, params = gym.make("CartPole-v1")
env = EpisodeStatistics(
    TimeLimit(GymnaxToLeraxEnv(gymnax_env, params), max_episode_steps=512)
)
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()

policy = algo.learn(
    env,
    policy,
    total_timesteps=2**16,
    key=learn_key,
    show_progress_bar=True,
    tb_log=True,
)
