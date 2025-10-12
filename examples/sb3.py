import equinox as eqx
from stable_baselines3 import PPO

from lerax.compatibility.gym import LeraxEnv
from lerax.env import CartPole

env = LeraxEnv(*eqx.nn.make_with_state(CartPole)())
model = PPO("MlpPolicy", env, tensorboard_log="logs")
model.learn(total_timesteps=2**16, progress_bar=True)
