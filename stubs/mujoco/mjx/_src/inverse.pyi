"""
Inverse dynamics functions.
"""

from __future__ import annotations

import mujoco.mjx._src.types
from jax import numpy as jp
from mujoco.mjx._src import derivative, forward, sensor, smooth, solver, support
from mujoco.mjx._src.types import Data, DisableBit, EnableBit, IntegratorType, Model

__all__: list[str] = [
    "Data",
    "DisableBit",
    "EnableBit",
    "IntegratorType",
    "Model",
    "derivative",
    "discrete_acc",
    "forward",
    "inv_constraint",
    "inverse",
    "jp",
    "sensor",
    "smooth",
    "solver",
    "support",
]

def discrete_acc(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Convert discrete-time qacc to continuous-time qacc.
    """

def inv_constraint(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Inverse constraint solver.
    """

def inverse(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Inverse dynamics.
    """
