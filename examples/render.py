from jax import lax
from jax import random as jr

from lerax.algorithm import PPO
from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy
from lerax.wrapper import TimeLimit

key = jr.key(0)
key, policy_key, learn_key = jr.split(key, 3)

env = TimeLimit(CartPole(renderer="auto"), max_episode_steps=512)
assert env.unwrapped.renderer is not None
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()
policy = algo.learn(env, policy, total_timesteps=2**14, key=learn_key)


def step(env_state, key):
    observation_key, action_key, transition_key, terminal_key, reset_key = jr.split(
        key, 5
    )

    action = policy(env.observation(env_state, key=observation_key), key=action_key)
    env_state = env.transition(env_state, action, key=transition_key)
    termination = env.terminal(env_state, key=terminal_key)
    truncation = env.truncate(env_state)

    env_state = lax.cond(
        termination | truncation,
        lambda: env.reset(key=reset_key)[0],
        lambda: env_state,
    )

    return env_state, env_state


key, reset_key = jr.split(key)
env_state = env.initial(key=reset_key)

_, env_states = lax.scan(
    step,
    env_state,
    jr.split(key, 512),
)

renderer = env.unwrapped.renderer
assert renderer is not None
renderer.open()
env.render_stacked(env_states, dt=1 / 60)  # pyright: ignore
renderer.close()
