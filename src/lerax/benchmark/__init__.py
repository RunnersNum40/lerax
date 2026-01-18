from __future__ import annotations

import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicy, AbstractPolicyState


def rollout_while(
    env: AbstractEnvLike,
    policy: AbstractPolicy,
    *,
    key: Key,
    deterministic: bool = False,
) -> Float[Array, ""]:
    def continue_rollout(
        carry: tuple[AbstractEnvLikeState, AbstractPolicyState, Key, Float[Array, ""]],
    ) -> Bool[Array, ""]:
        env_state, _, key, _ = carry
        return ~(env.terminal(env_state, key=key) | env.truncate(env_state))

    def step(
        carry: tuple[AbstractEnvLikeState, AbstractPolicyState, Key, Float[Array, ""]],
    ):
        env_state, policy_state, key, cumulative_reward = carry
        carry_key, obs_key, action_key, transition_key = jr.split(key, 4)

        obs = env.observation(env_state, key=obs_key)

        if deterministic:
            next_policy_state, action = policy(policy_state, obs)
        else:
            next_policy_state, action = policy(policy_state, obs, key=action_key)

        next_env_state = env.transition(env_state, action, key=transition_key)
        reward = env.reward(env_state, action, next_env_state, key=carry_key)

        return (
            next_env_state,
            next_policy_state,
            carry_key,
            cumulative_reward + reward,
        )

    env_state = env.initial(key=key)
    policy_state = policy.reset(key=key)
    carry = (env_state, policy_state, key, jnp.array(0.0))
    carry = jax.lax.while_loop(continue_rollout, step, carry)
    return carry[3]


def rollout_scan(
    env: AbstractEnvLike,
    policy: AbstractPolicy,
    *,
    key: Key,
    deterministic: bool = False,
    max_steps: int = 1024,
) -> Float[Array, ""]:

    def step(
        carry: tuple[AbstractEnvLikeState, AbstractPolicyState, Bool[Array, ""]],
        key: Key,
    ) -> tuple[
        tuple[AbstractEnvLikeState, AbstractPolicyState, Bool[Array, ""]],
        Float[Array, ""],
    ]:
        env_state, policy_state, done = carry

        def next_step():
            carry_key, obs_key, action_key, transition_key, terminal_key = jr.split(
                key, 5
            )

            obs = env.observation(env_state, key=obs_key)

            if deterministic:
                next_policy_state, action = policy(policy_state, obs)
            else:
                next_policy_state, action = policy(policy_state, obs, key=action_key)

            next_env_state = env.transition(env_state, action, key=transition_key)
            reward = env.reward(env_state, action, next_env_state, key=carry_key)
            done = env.terminal(next_env_state, key=terminal_key) | env.truncate(
                next_env_state
            )

            return (next_env_state, next_policy_state, done), reward

        def done_step():
            return (env_state, policy_state, jnp.array(True)), jnp.array(0.0)

        return jax.lax.cond(done, done_step, next_step)

    env_state = env.initial(key=key)
    policy_state = policy.reset(key=key)
    carry = (env_state, policy_state, jnp.array(False))
    _, rewards = jax.lax.scan(step, carry, jr.split(key, max_steps))
    return jnp.sum(rewards)


def average_reward(
    env: AbstractEnvLike,
    policy: AbstractPolicy,
    num_episodes: int = 4,
    max_steps: int | None = 1024,
    deterministic: bool = False,
    *,
    key: Key,
) -> Float[Array, ""]:
    def episode_reward(key: Key) -> Float[Array, ""]:
        if max_steps is None:
            return rollout_while(env, policy, key=key, deterministic=deterministic)
        else:
            return rollout_scan(
                env, policy, key=key, deterministic=deterministic, max_steps=max_steps
            )

    rewards = jax.vmap(episode_reward)(jr.split(key, num_episodes))
    return jnp.mean(rewards)
