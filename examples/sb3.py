from stable_baselines3 import PPO

from lerax.compatibility.gym import LeraxToGymEnv
from lerax.env.classic_control import CartPole

env = LeraxToGymEnv(CartPole())
model = PPO("MlpPolicy", env, tensorboard_log="logs")
model.learn(total_timesteps=2**16, progress_bar=True)
