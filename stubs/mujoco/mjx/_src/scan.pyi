"""
Scan across data ordered by body joint types and kinematic tree order.
"""

from __future__ import annotations

import typing
from typing import Any, TypeVar

import jax as jax
import mujoco.mjx._src.types
import numpy
import numpy as np
from jax import numpy as jp
from mujoco.mjx._src.types import JointType, Model, TrnType

__all__: list[str] = [
    "Any",
    "JointType",
    "Model",
    "TrnType",
    "TypeVar",
    "Y",
    "body_tree",
    "flat",
    "jax",
    "jp",
    "np",
]

def _check_input(
    m: mujoco.mjx._src.types.Model, args: typing.Any, in_types: str
) -> None:
    """
    Checks that scan input has the right shape.
    """

def _check_output(y: jax.Array, take_ids: numpy.ndarray, typ: str, idx: int) -> None:
    """
    Checks that scan output has the right shape.
    """

def _index(haystack: numpy.ndarray, needle: numpy.ndarray) -> numpy.ndarray:
    """
    Returns indexes in haystack for elements in needle.
    """

def _nvmap(f: typing.Callable[..., ~Y], *args) -> ~Y:
    """
    A vmap that accepts numpy arrays.

    Numpy arrays are statically vmapped, and the elements are passed to f as
    static arguments.  The implication is that all the elements of numpy array
    arguments must be the same.

    Args:
      f: function to be mapped over
      *args: args to be mapped along, passed to f

    Returns:
      the result of vmapping f over args

    Raises:
      RuntimeError: if numpy arg elements do not match
    """

def _q_bodyid(m: mujoco.mjx._src.types.Model) -> numpy.ndarray:
    """
    Returns the bodyid for each qpos adress.
    """

def _q_jointid(m: mujoco.mjx._src.types.Model) -> numpy.ndarray:
    """
    Returns the jointid for each qpos adress.
    """

def _take(obj: ~Y, idx: numpy.ndarray) -> ~Y:
    """
    Takes idxs on any pytree given to it.

    XLA executes x[jp.array([1, 2, 3])] slower than x[1:4], so we detect when
    take indices are contiguous, and convert them to slices.

    Args:
      obj: an input pytree
      idx: indices to take

    Returns:
      obj pytree with leaves taken by idxs
    """

def body_tree(
    m: mujoco.mjx._src.types.Model,
    f: typing.Callable[..., ~Y],
    in_types: str,
    out_types: str,
    *args,
    reverse: bool = False,
) -> ~Y:
    """
    Scan ``f`` across bodies in tree order, carrying results up/down the tree.

    This function groups bodies according to level and attached joints, then calls
    vmap(f) on them.\\

    Args:
      m: an mjx mjmodel
      f: a function to be scanned with the following type signature:
          def f(y, *args) -> y
        where
          ``y`` is the carry value and return value
          ``*args`` are input arguments with types matching ``in_types``
      in_types: string specifying the type of each input arg:
        'b': split according to bodies
        'j': split according to joint types
        'q': split according to generalized coordinates (len(qpos))
        'v': split according to degrees of freedom (len(qvel))
      out_types: string specifying the types the output dimension matches
      *args: the input arguments corresponding to ``in_types``
      reverse: if True, scans up the body tree from leaves to root, otherwise
        root to leaves

    Returns:
      The stacked outputs of ``f`` matching the model's body order.

    Raises:
        IndexError: if function output shape does not match out_types shape
    """

def flat(
    m: mujoco.mjx._src.types.Model,
    f: typing.Callable[..., ~Y],
    in_types: str,
    out_types: str,
    *args,
    group_by: str = "j",
) -> ~Y:
    """
    Scan a function across bodies or actuators.

    Scan group data according to type and batch shape then calls vmap(f) on it.\\

    Args:
      m: an mjx model
      f: a function to be scanned with the following type signature:
          def f(key, *args) -> y
        where
          ``key`` gives grouping key for this function instance
          ``*args`` are input arguments with types matching ``in_types``
          ``y`` is an output arguments with types matching ``out_type``
      in_types: string specifying the type of each input arg:
        'b': split according to bodies
        'j': split according to joint types
        'q': split according to generalized coordinates (len(qpos))
        'v': split according to degrees of freedom (len(qvel))
        'u': split according to actuators
        'a': split according to actuator activations
        'c': split according to camera
      out_types: string specifying the types the output dimension matches
      *args: the input arguments corresponding to ``in_types``
      group_by: the type to group by, either joints or actuators

    Returns:
      The stacked outputs of ``f`` matching the model's order.

    Raises:
        IndexError: if function output shape does not match out_types shape
    """

Y: typing.TypeVar  # value = ~Y
