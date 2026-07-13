from __future__ import annotations

import equinox as eqx

from lerax.utils import polyak_average

from .base_target import AbstractTargetUpdate


class SoftUpdate(AbstractTargetUpdate):
    """Polyak averaging

    target = tau * online + (1 - tau) * target.

    Args:
        tau: Interpolation coefficient in [0, 1]. Default 0.005.

    Attributes:
        tau: Interpolation coefficient in [0, 1].
    """

    tau: float = 0.005

    def __call__[T: eqx.Module](self, online: T, target: T) -> T:
        return polyak_average(online, target, self.tau)
