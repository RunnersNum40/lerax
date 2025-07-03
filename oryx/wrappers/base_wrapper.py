import equinox as eqx
from jaxtyping import PyTree

from oryx.env import AbstractEnv


class AbstractWrapper[WrapperActType, WrapperObsType, ActType, ObsType](
    AbstractEnv[WrapperActType, WrapperObsType], strict=True
):
    """Base class for environment wrappers"""

    env: eqx.AbstractVar[AbstractEnv[ActType, ObsType]]

    @property
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        return self.env.unwrapped
