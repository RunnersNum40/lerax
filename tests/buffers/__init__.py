from jax import numpy as jnp

from oryx.buffer import RolloutBuffer


def empty_buffer(n: int = 8):
    observations = jnp.zeros((n,))
    actions = jnp.zeros((n,))
    rewards = jnp.zeros((n,))
    terminations = jnp.full((n,), False)
    truncations = jnp.full((n,), False)
    log_probs = jnp.zeros((n,))
    values = jnp.zeros((n,))
    states = jnp.zeros((n,))
    returns = jnp.zeros((n,))
    advantages = jnp.ones((n,))

    return RolloutBuffer(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        log_probs=log_probs,
        values=values,
        states=states,
        returns=returns,
        advantages=advantages,
    )
