from __future__ import annotations

from abc import abstractmethod

import equinox as eqx


class AbstractTargetUpdate(eqx.Module):
    """Rule for refreshing a target network from an online network."""

    @abstractmethod
    def __call__[T: eqx.Module](self, online: T, target: T) -> T:
        """Return the updated target network."""
