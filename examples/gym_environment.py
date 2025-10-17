import gymnasium as gym
from jax import random as jr

from lerax.algorithm import PPO
from lerax.compatibility.gym import GymToLeraxEnv
from lerax.policy import MLPActorCriticPolicy
from lerax.wrapper import EpisodeStatistics, TimeLimit

policy_key, learn_key = jr.split(jr.key(0), 2)

gym_env = gym.make("CartPole-v1")
env = EpisodeStatistics(TimeLimit(GymToLeraxEnv(gym_env), max_episode_steps=512))
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
