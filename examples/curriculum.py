"""Curriculum learning example: gradually increase pendulum mass during PPO training."""

from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import CallbackList, ConsoleBackend, LoggingCallback
from lerax.curriculum import ScheduledCurriculum, linear_schedule
from lerax.env.classic_control import Pendulum
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = Pendulum()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO(num_envs=4, num_steps=2048)

# Schedule pendulum mass from 0.5 to 2.0 over 500 iterations
curriculum = ScheduledCurriculum(
    where=lambda env: env.m,
    schedule_fn=linear_schedule(start=0.5, end=2.0, total=500),
)

logger = LoggingCallback([ConsoleBackend()], env=env, policy=policy)
callbacks = CallbackList(callbacks=[curriculum, logger])

policy = algo.learn(
    env, policy, total_timesteps=2**20, key=learn_key, callback=callbacks
)

print(f"Final mass: {jnp.array(2.0)}")
logger.close()
