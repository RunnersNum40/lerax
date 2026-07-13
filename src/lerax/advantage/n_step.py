from __future__ import annotations

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from .base_advantage import AbstractAdvantageEstimator


class NStepReturn(AbstractAdvantageEstimator):
    """N-step bootstrapped returns.

    Args:
        n: Maximum number of rewards accumulated before bootstrapping.
        gamma: Discount factor.

    Attributes:
        n: Maximum number of rewards accumulated before bootstrapping.
        gamma: Discount factor.
    """

    n: int = eqx.field(static=True)
    gamma: Float[ArrayLike, ""] = 0.99

    def __init__(self, n: int, gamma: Float[ArrayLike, ""] = 0.99):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n
        self.gamma = gamma

    def __call__(
        self,
        rewards: Float[Array, " T"],
        values: Float[Array, " T"],
        dones: Bool[Array, " T"],
        last_value: Float[ArrayLike, ""],
    ) -> tuple[Float[Array, " T"], Float[Array, " T"]]:
        gamma = jnp.asarray(self.gamma)
        last_value = jnp.asarray(last_value)
        trajectory_length = rewards.shape[0]

        discounts = gamma * (1.0 - dones.astype(float))
        padded_rewards = jnp.concatenate(
            [rewards, jnp.zeros(self.n, dtype=rewards.dtype)]
        )
        padded_discounts = jnp.concatenate(
            [discounts, jnp.ones(self.n, dtype=discounts.dtype)]
        )
        bootstrap_values = jnp.concatenate(
            [values, jnp.full((self.n,), last_value, dtype=values.dtype)]
        )[self.n : self.n + trajectory_length]

        def accumulate(
            carry: Float[Array, " T"],
            offset: Integer[Array, ""],
        ) -> tuple[Float[Array, " T"], None]:
            reward = lax.dynamic_slice_in_dim(padded_rewards, offset, trajectory_length)
            discount = lax.dynamic_slice_in_dim(
                padded_discounts, offset, trajectory_length
            )
            return reward + discount * carry, None

        returns, _ = lax.scan(
            accumulate, bootstrap_values, jnp.arange(self.n), reverse=True
        )
        return returns - values, returns
