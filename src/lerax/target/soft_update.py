from __future__ import annotations

import equinox as eqx

from lerax.utils import polyak_average

from .base_target import AbstractTargetUpdate


class SoftUpdate(AbstractTargetUpdate):
    """Apply Polyak averaging.

    target = tau * online + (1 - tau) * target.

    Attributes:
        tau: Interpolation coefficient in [0, 1]. Defaults to 0.005.
    """

    tau: float = 0.005

    def __call__[T: eqx.Module](self, online: T, target: T) -> T:
        return polyak_average(online, target, self.tau)
