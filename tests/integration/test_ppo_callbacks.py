import os
from tempfile import TemporaryDirectory

from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ProgressBarCallback, TensorBoardCallback
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy


def test_ppo_with_callbacks():
    """Test PPO training with ProgressBar and TensorBoard callbacks."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = PPO()

    total_timesteps = 512
    progress_bar = ProgressBarCallback(total_timesteps)
    directory = TemporaryDirectory()
    tensorboard = TensorBoardCallback(log_dir=directory.name, env=env, policy=policy)
    callbacks = [progress_bar, tensorboard]

    trained_policy = algo.learn(
        env, policy, total_timesteps=total_timesteps, key=learn_key, callback=callbacks
    )

    assert trained_policy is not None
    assert os.path.exists(directory.name)
    assert len(os.listdir(directory.name)) > 0
