from __future__ import annotations

from abc import abstractmethod
from typing import Concatenate

import equinox as eqx
from jaxtyping import Key


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


class AbstractStochasticModel[**InType, OutType](
    AbstractModel[
        Concatenate[Key, InType],
        OutType,
    ],
):
    """Base class for stochastic models that take inputs and produce outputs."""

    # TODO: Revisit this if Python ever adds support for concatenating keyword
    # parameters, I'd rather key as a keyword to fit the style of the library
    @abstractmethod
    def __call__(
        self, key: Key, *args: InType.args, **kwargs: InType.kwargs
    ) -> OutType:
        """Return an output given an input."""


class AbstractStochasticStatefulModel[**InType, *OutType](
    AbstractStatefulModel[
        Concatenate[Key, InType],
        *OutType,
    ],
    AbstractStochasticModel[
        Concatenate[eqx.nn.State, InType],
        tuple[eqx.nn.State, *OutType],
    ],
):
    """Base class for stochastic models with state."""

    @abstractmethod
    def __call__(
        self, state: eqx.nn.State, key: Key, *args: InType.args, **kwargs: InType.kwargs
    ) -> tuple[eqx.nn.State, *OutType]:
        """Return an output given inputs and the state."""
