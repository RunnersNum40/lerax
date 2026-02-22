"""
Sensor functions.
"""

from __future__ import annotations

import jax as jax
import mujoco as mujoco
import numpy as np
from jax import numpy as jp
from mujoco.mjx._src import math, ray, smooth, support
from mujoco.mjx._src.types import (
    Data,
    DataJAX,
    DisableBit,
    Model,
    ModelJAX,
    ObjType,
    SensorType,
    TrnType,
)

__all__: list[str] = [
    "Data",
    "DataJAX",
    "DisableBit",
    "Model",
    "ModelJAX",
    "ObjType",
    "SensorType",
    "TrnType",
    "jax",
    "jp",
    "math",
    "mujoco",
    "np",
    "ray",
    "sensor_acc",
    "sensor_pos",
    "sensor_vel",
    "smooth",
    "support",
]

def _apply_cutoff(sensor: jax.Array, cutoff: jax.Array, data_type: int) -> jax.Array:
    """
    Clip sensor to cutoff value.
    """

def sensor_acc(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Compute acceleration/force-dependent sensors values.
    """

def sensor_pos(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Compute position-dependent sensors values.
    """

def sensor_vel(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Compute velocity-dependent sensors values.
    """
