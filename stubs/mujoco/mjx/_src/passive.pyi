"""
Passive forces.
"""

from __future__ import annotations

import jax as jax
import mujoco.mjx._src.types
from jax import numpy as jp
from mujoco.mjx._src import math, scan, support
from mujoco.mjx._src.types import (
    Data,
    DataJAX,
    DisableBit,
    JointType,
    Model,
    ModelJAX,
    OptionJAX,
)

__all__: list[str] = [
    "Data",
    "DataJAX",
    "DisableBit",
    "JointType",
    "Model",
    "ModelJAX",
    "OptionJAX",
    "jax",
    "jp",
    "math",
    "passive",
    "scan",
    "support",
]

def _fluid(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> jax.Array:
    """
    Applies body-level viscosity, lift and drag.
    """

def _gravcomp(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> jax.Array:
    """
    Applies body-level gravity compensation.
    """

def _inertia_box_fluid_model(
    m: mujoco.mjx._src.types.Model,
    inertia: jax.Array,
    mass: jax.Array,
    root_com: jax.Array,
    xipos: jax.Array,
    ximat: jax.Array,
    cvel: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Fluid forces based on inertia-box approximation.
    """

def _spring_damper(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> jax.Array:
    """
    Applies joint level spring and damping forces.
    """

def passive(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Adds all passive forces.
    """
