from __future__ import annotations

from abc import abstractmethod
from typing import Concatenate

import equinox as eqx


class AbstractModel[**InType, OutType](eqx.Module):
    """Base class for models that take inputs and produce outputs."""

    @abstractmethod
    def __call__(self, *args: InType.args, **kwargs: InType.kwargs) -> OutType:
        """Return an output given an input."""


class AbstractStatefulModel[**InType, *OutType](
    AbstractModel[Concatenate[eqx.nn.State, InType], tuple[eqx.nn.State, *OutType]],
):
    """Base class for models with state."""

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]

    @abstractmethod
    def __call__(
        self, state: eqx.nn.State, *args: InType.args, **kwargs: InType.kwargs
    ) -> tuple[eqx.nn.State, *OutType]:
        """Return an output given inputs and the state."""

    @abstractmethod
    def reset(self, state: eqx.nn.State) -> eqx.nn.State:
        """Reset the state of the model to its initial state."""
