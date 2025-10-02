import equinox as eqx
from stable_baselines3 import PPO

from oryx.compatibility.gym import OryxEnv
from oryx.env import CartPole

env = OryxEnv(*eqx.nn.make_with_state(CartPole)())
model = PPO("MlpPolicy", env, tensorboard_log="logs")
model.learn(total_timesteps=2**17, progress_bar=True)
