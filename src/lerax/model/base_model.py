from __future__ import annotations

from abc import abstractmethod
from typing import Concatenate

import equinox as eqx


class AbstractModelState(eqx.Module):
    pass


class AbstractModel[**InType, OutType](eqx.Module):
    """Base class for models that take inputs and produce outputs."""

    @abstractmethod
    def __call__(self, *args: InType.args, **kwargs: InType.kwargs) -> OutType:
        """Return an output given an input."""


class AbstractStatefulModel[StateType: AbstractModelState, **InType, *OutType](
    AbstractModel[Concatenate[StateType, InType], tuple[StateType, *OutType]],
):
    """Base class for models with state."""

    @abstractmethod
    def reset(self) -> StateType:
        """Reset the model to its initial state."""
