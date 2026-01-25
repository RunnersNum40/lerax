import tempfile

from stable_baselines3 import PPO

from lerax.compatibility.gym import LeraxToGymEnv
from lerax.env import CartPole


def test_sb3_integration():
    """Test training SB3 PPO on a Lerax environment."""
    env = LeraxToGymEnv(CartPole())

    with tempfile.TemporaryDirectory() as tmpdir:
        model = PPO("MlpPolicy", env, tensorboard_log=tmpdir, verbose=0)
        model.learn(total_timesteps=512, progress_bar=False)

    assert model is not None
