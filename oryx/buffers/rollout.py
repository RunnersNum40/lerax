from __future__ import annotations

from jaxtyping import Array

from oryx.spaces import AbstractSpace

from .base_buffer import AbstractBuffer


class RolloutBuffer(AbstractBuffer, strict=True):

    observation_space: AbstractSpace
    observations: Array
    actions: Array
    rewards: Array
    advantages: Array
    returns: Array
    episode_starts: Array
    log_probs: Array
    values: Array

    def reset(self) -> RolloutBuffer:
        pass

    def compute_returns_and_advantages(
        self, last_value: Array, dones: Array
    ) -> RolloutBuffer:
        pass
