from __future__ import annotations

import equinox as eqx

from .base_target import AbstractTargetUpdate


class HardUpdate(AbstractTargetUpdate):
    """Hard copy."""

    def __call__[T: eqx.Module](self, online: T, target: T) -> T:
        return online
