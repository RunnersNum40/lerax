import os
from tempfile import TemporaryDirectory

from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import ConsoleBackend, LoggingCallback, TensorBoardBackend
from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy


def test_ppo_with_callbacks():
    """Test PPO training with console and TensorBoard logging backends."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)
    algo = PPO()

    total_timesteps = 512
    directory = TemporaryDirectory()
    logger = LoggingCallback(
        [
            TensorBoardBackend(log_dir=directory.name),
            ConsoleBackend(total_timesteps=total_timesteps),
        ],
        env=env,
        policy=policy,
    )

    trained_policy = algo.learn(
        env, policy, total_timesteps=total_timesteps, key=learn_key, callback=logger
    )
    logger.close()

    assert trained_policy is not None
    assert os.path.exists(directory.name)
    assert len(os.listdir(directory.name)) > 0
