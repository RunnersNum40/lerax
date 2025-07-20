from abc import abstractmethod
from typing import Concatenate

import equinox as eqx


class AbstractModel[**P, T](eqx.Module, strict=True):
    """Base class for models that take inputs and produce outputs."""

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Return an output given an input."""


class AbstractStatefulModel[**P, *T](
    AbstractModel[Concatenate[eqx.nn.State, P], tuple[eqx.nn.State, *T]],
    strict=True,
):
    """Base class for models with state."""

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]

    @abstractmethod
    def __call__(
        self, state: eqx.nn.State, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[eqx.nn.State, *T]:
        """Return an output given inputs and the state."""
