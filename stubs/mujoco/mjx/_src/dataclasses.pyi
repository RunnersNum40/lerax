"""
Wrapper that automatically registers dataclass as a Jax PyTree.
"""

from __future__ import annotations

import copy as copy
import dataclasses as dataclasses
import typing as typing
import warnings as warnings
from typing import TypeVar

import jax as jax
import numpy as np

__all__: list[str] = [
    "PyTreeNode",
    "TNode",
    "TypeVar",
    "copy",
    "dataclass",
    "dataclasses",
    "jax",
    "np",
    "typing",
    "warnings",
]

class PyTreeNode:
    """
    Base class for dataclasses that should act like a JAX pytree node.

    This base class additionally avoids type checking errors when using PyType.
    """
    @classmethod
    def __init_subclass__(cls, register_as_pytree: bool = True, **kwargs): ...
    @classmethod
    def fields(cls) -> typing.Tuple[dataclasses.Field, ...]: ...
    def __init__(self, *args, **kwargs): ...
    def replace(self: ~TNode, **overrides) -> ~TNode: ...
    def tree_replace(
        self,
        params: typing.Dict[
            str,
            typing.Union[
                jax.Array,
                numpy.ndarray,
                numpy.bool,
                numpy.number,
                bool,
                int,
                float,
                complex,
                jax._src.literals.TypedNdArray,
                NoneType,
            ],
        ],
    ) -> PyTreeNode: ...

def _jax_in_args(typ) -> bool: ...
def _tree_replace(
    base: PyTreeNode,
    attr: typing.Sequence[str],
    val: typing.Union[
        jax.Array,
        numpy.ndarray,
        numpy.bool,
        numpy.number,
        bool,
        int,
        float,
        complex,
        jax._src.literals.TypedNdArray,
        NoneType,
    ],
) -> PyTreeNode:
    """
    Sets attributes in a struct.dataclass with values.
    """

def dataclass(clz: ~_T, register_as_pytree: bool) -> ~_T:
    """
    Wraps a dataclass with metadata for which fields are pytrees.

    This is based off flax.struct.dataclass, but instead of using field
    descriptors to specify which fields are pytrees, we follow a simple rule:
    a leaf field is a pytree node if and only if it's a jax.Array

    Args:
      clz: the class to register as a dataclass

    Returns:
      the resulting dataclass, registered with Jax
    """

TNode: typing.TypeVar  # value = ~TNode
_T: typing.TypeVar  # value = ~_T
