from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, ArrayLike, Bool, Float


class AbstractAdvantageEstimator(eqx.Module):
    """Estimates advantages and returns from a trajectory segment.

    Attributes:
        gamma: Discount factor.
    """

    gamma: eqx.AbstractVar[Float[ArrayLike, ""]]

    @abstractmethod
    def __call__(
        self,
        rewards: Float[Array, " T"],
        values: Float[Array, " T"],
        dones: Bool[Array, " T"],
        last_value: Float[ArrayLike, ""],
    ) -> tuple[Float[Array, " T"], Float[Array, " T"]]:
        """Compute advantages and returns.

        Args:
            rewards: Per-step rewards.
            values: Per-step value estimates.
            dones: Whether each step ended an episode.
            last_value: Bootstrap value after the trajectory segment. Strict
                episodic estimators may ignore it.

        Returns:
            The advantages and returns, each with shape ``(T,)``.
        """
