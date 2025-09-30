from datetime import datetime

import equinox as eqx
from jax import random as jr

from oryx.algorithm import PPO
from oryx.env import CartPole
from oryx.policy import MLPActorCriticPolicy
from oryx.wrapper import EpisodeStatistics, TimeLimit

policy_key, learn_key = jr.split(jr.key(0), 2)

env = EpisodeStatistics(TimeLimit(CartPole(), max_episode_steps=512))
policy = policy = MLPActorCriticPolicy(
    env=env,
    key=policy_key,
)

algo, state = eqx.nn.make_with_state(PPO)(
    env=env,
    policy=policy,
)

algo.learn(
    state,
    total_timesteps=2**18,
    key=learn_key,
    show_progress_bar=True,
    tb_log_name=f"logs/{env.name}_{datetime.now().strftime("%H%M%S")}",
)
