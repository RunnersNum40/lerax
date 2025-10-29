import time

from jax import lax
from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import PPO
from lerax.env import AbstractEnvLikeState, CartPole
from lerax.policy import MLPActorCriticPolicy
from lerax.utils import unstack_pytree
from lerax.wrapper import EpisodeStatistics, TimeLimit

key = jr.key(0)
key, policy_key, learn_key = jr.split(key, 3)

env = EpisodeStatistics(TimeLimit(CartPole(renderer="auto"), max_episode_steps=512))
assert env.unwrapped.renderer is not None
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()
policy = algo.learn(
    env,
    policy,
    total_timesteps=2**14,
    key=learn_key,
)


def step(carry: tuple, _) -> tuple[tuple, AbstractEnvLikeState]:
    key, env_state, observation = carry
    key, action_key, step_key, reset_key = jr.split(key, 4)

    action = policy(observation, key=action_key)
    env_state, observation, _, termination, truncation, _ = env.step(
        env_state, action, key=step_key
    )

    env_state = lax.cond(
        jnp.logical_or(termination, truncation),
        lambda: env.reset(key=reset_key)[0],
        lambda: env_state,
    )

    return (key, env_state, observation), env_state


key, reset_key = jr.split(key)
env_state, observation, _ = env.reset(key=reset_key)

_, env_states = lax.scan(
    step,
    (key, env_state, observation),
    length=512,
)

renderer = env.unwrapped.renderer
assert renderer is not None
renderer.open()
for s in unstack_pytree(env_states):
    env.render(s)  # pyright: ignore
    time.sleep(1 / 64)
renderer.close()
