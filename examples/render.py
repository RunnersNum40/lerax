from jax import lax
from jax import random as jr

from lerax.env import CartPole

env = CartPole()


def step(env_state, key):
    action_key, transition_key, terminal_key, reset_key = jr.split(key, 4)

    action = env.action_space.sample(key=action_key)
    env_state = env.transition(env_state, action, key=transition_key)
    done = env.terminal(env_state, key=terminal_key) | env.truncate(env_state)

    env_state = lax.cond(
        done,
        lambda: env.initial(key=reset_key),
        lambda: env_state,
    )

    return env_state, env_state


reset_key, rollout_key = jr.split(jr.key(0), 2)

_, env_states = lax.scan(step, env.initial(key=reset_key), jr.split(rollout_key, 1024))

env.render_stacked(env_states, dt=1 / 60)
