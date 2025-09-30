import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Key

from oryx.algorithm import PPO
from oryx.env import AbstractEnvLike, CartPole
from oryx.model import MLP
from oryx.policy import CustomActorCriticPolicy
from oryx.wrapper import EpisodeStatistics, TimeLimit


def build_env():
    return EpisodeStatistics(TimeLimit(CartPole(), max_episode_steps=512))


def build_policy(env: AbstractEnvLike, *, key: Key) -> CustomActorCriticPolicy:
    mlp_key, pol_key = jr.split(key, 2)

    obs_dim = int(jnp.prod(jnp.asarray(env.observation_space.shape)))
    n_actions = int(env.action_space.n)
    action_head = MLP(
        in_size=obs_dim,
        out_size=n_actions,
        width_size=64,
        depth=2,
        key=mlp_key,
    )
    policy = CustomActorCriticPolicy(
        env=env,
        action_model=action_head,
        key=pol_key,
    )
    return policy


def main():
    seed = 0
    total_timesteps = 2**20
    logdir = "logs/CartPole_PPO"

    policy_key, learn_key = jr.split(jr.key(seed), 2)

    env = build_env()
    policy = build_policy(env, key=policy_key)

    algo, state = eqx.nn.make_with_state(PPO)(
        env=env,
        policy=policy,
    )

    algo.learn(
        state,
        total_timesteps=total_timesteps,
        key=learn_key,
        show_progress_bar=True,
        tb_log_name=logdir,
    )


if __name__ == "__main__":
    main()
