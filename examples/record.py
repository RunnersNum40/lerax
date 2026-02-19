from jax import lax
from jax import random as jr

from lerax.algorithm import PPO
from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy
from lerax.render import VideoRenderer

key = jr.key(0)
key, policy_key, learn_key = jr.split(key, 3)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)
algo = PPO()
policy = algo.learn(env, policy, total_timesteps=2**16, key=learn_key)


def step(carry, key):
    observation_key, action_key, transition_key, terminal_key, reset_key = jr.split(
        key, 5
    )

    env_state, policy_state = carry

    observation = env.observation(env_state, key=observation_key)
    policy_state, action = policy(policy_state, observation, key=action_key)
    env_state = env.transition(env_state, action, key=transition_key)
    termination = env.terminal(env_state, key=terminal_key)
    truncation = env.truncate(env_state)

    env_state, policy_state = lax.cond(
        termination | truncation,
        lambda: (env.initial(key=reset_key), policy.reset(key=reset_key)),
        lambda: (env_state, policy_state),
    )

    return (env_state, policy_state), env_state


key, reset_key = jr.split(key)
env_state = env.initial(key=reset_key)
policy_state = policy.reset(key=reset_key)

_, env_states = lax.scan(step, (env_state, policy_state), jr.split(key, 1024))

renderer = VideoRenderer(
    inner=env.default_renderer(),
    output_path="cartpole.mp4",
    fps=60.0,
)

env.render_stacked(env_states, renderer=renderer, dt=1 / 60)
