from __future__ import annotations

from abc import abstractmethod

import equinox as eqx


class AbstractBuffer(eqx.Module, strict=True):
    """Base class for buffers."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the buffer."""
