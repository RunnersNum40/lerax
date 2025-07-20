import equinox as eqx

from oryx.buffers import RolloutBuffer
from oryx.policies import AbstractPolicy

from .base_algorithm import AbstractAlgorithm


class AbstractOnPolicyAlgorithm(AbstractAlgorithm, strict=True):
    """Base class for on policy algorithms."""

    rollout_buffer: eqx.AbstractVar[RolloutBuffer]
    policy: eqx.AbstractVar[AbstractPolicy]
