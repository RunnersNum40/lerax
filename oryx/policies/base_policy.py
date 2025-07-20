from abc import abstractmethod

import equinox as eqx
from jaxtyping import Key


class AbstractPolicy[ActType, ObsType](eqx.Module, strict=True):
    """
    Base class for policies.

    Policies map from observations to actions.
    """

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]

    @abstractmethod
    def predict(
        self, state: eqx.nn.State, observation: ObsType, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, ActType]:
        """Choose an action from an observation."""
