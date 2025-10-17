import time

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr

from lerax.env import CartPole

key = jr.key(0)
dt = 1 / 16

env, state = eqx.nn.make_with_state(CartPole)(renderer="auto")
assert env.renderer is not None
env.renderer.open()

key, reset_key = jr.split(key)
state, _, _ = env.reset(key=reset_key)

for _ in range(256):
    key, action_key, step_key, reset_key = jr.split(key, 4)

    action = env.action_space.sample(action_key)
    state, _, _, termination, truncation, _ = env.step(state, action, key=step_key)
    if jnp.logical_or(termination, truncation):
        state = env.reset(key=reset_key)[0]

    env.render(state)
    time.sleep(dt)

env.renderer.close()
