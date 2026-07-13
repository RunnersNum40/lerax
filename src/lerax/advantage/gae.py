from __future__ import annotations

from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from .base_advantage import AbstractAdvantageEstimator


class GAE(AbstractAdvantageEstimator):
    """Generalized Advantage Estimation.

    Args:
        gamma: Discount factor
        lam: Bias-variance tradeoff

    Attributes:
        gamma: Discount factor
        lam: Bias-variance tradeoff
    """

    gamma: Float[ArrayLike, ""] = 0.99
    lam: Float[ArrayLike, ""] = 0.95

    def __init__(
        self,
        gamma: Float[ArrayLike, ""] = 0.99,
        lam: Float[ArrayLike, ""] = 0.95,
    ):
        self.gamma = gamma
        self.lam = lam

    def __call__(
        self,
        rewards: Float[Array, " T"],
        values: Float[Array, " T"],
        dones: Bool[Array, " T"],
        last_value: Float[ArrayLike, ""],
    ) -> tuple[Float[Array, " T"], Float[Array, " T"]]:
        gamma = jnp.asarray(self.gamma)
        lam = jnp.asarray(self.lam)
        last_value = jnp.asarray(last_value)

        next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
        not_done = 1.0 - dones.astype(float)
        deltas = rewards + gamma * next_values * not_done - values
        discounts = gamma * lam * not_done

        def scan_fn(
            carry: Float[Array, ""],
            inputs: tuple[Float[Array, ""], Float[Array, ""]],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            delta, discount = inputs
            advantage = delta + discount * carry
            return advantage, advantage

        _, advantages = lax.scan(
            scan_fn,
            jnp.zeros((), dtype=deltas.dtype),
            (deltas, discounts),
            reverse=True,
        )
        return advantages, advantages + values
