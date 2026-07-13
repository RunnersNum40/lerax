from __future__ import annotations

from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from .base_advantage import AbstractAdvantageEstimator


def discounted_returns(
    rewards: Float[Array, " T"],
    dones: Bool[Array, " T"],
    final_value: Float[ArrayLike, ""],
    gamma: Float[ArrayLike, ""],
) -> Float[Array, " T"]:
    """Compute discounted returns from a trajectory segment.

    Args:
        rewards: Per-step rewards
        dones: Whether each step ended an episode
        final_value: Return used to bootstrap the trajectory boundary
        gamma: Discount factor

    Returns:
        The discounted returns, with shape ``(T,)``.
    """
    gamma = jnp.asarray(gamma)
    final_value = jnp.asarray(final_value)
    not_done = 1.0 - dones.astype(float)

    def scan_fn(
        carry: Float[Array, ""],
        inputs: tuple[Float[Array, ""], Float[Array, ""]],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        reward, discount = inputs
        running_return = reward + gamma * discount * carry
        return running_return, running_return

    _, returns = lax.scan(
        scan_fn,
        final_value,
        (rewards, not_done),
        reverse=True,
    )
    return returns


class BootstrappedReturn(AbstractAdvantageEstimator):
    """Discounted returns bootstrapped at an incomplete segment boundary.

    Args:
        gamma: Discount factor.

    Attributes:
        gamma: Discount factor.
    """

    gamma: Float[ArrayLike, ""] = 0.99

    def __init__(self, gamma: Float[ArrayLike, ""] = 0.99):
        self.gamma = gamma

    def __call__(
        self,
        rewards: Float[Array, " T"],
        values: Float[Array, " T"],
        dones: Bool[Array, " T"],
        last_value: Float[ArrayLike, ""],
    ) -> tuple[Float[Array, " T"], Float[Array, " T"]]:
        returns = discounted_returns(
            rewards,
            dones,
            last_value,
            self.gamma,
        )
        return returns - values, returns
