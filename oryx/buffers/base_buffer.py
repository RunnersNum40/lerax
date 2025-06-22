import equinox as eqx

from oryx.spaces import AbstractSpace


class AbstractBuffer(eqx.Module, strict=True):
    """Base class for buffers."""

    observation_space: eqx.AbstractVar[AbstractSpace]
