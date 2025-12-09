"""
Derivative functions.
"""
from __future__ import annotations

import jax as jax
import mujoco.mjx._src.types
from jax import numpy as jp
from mujoco.mjx._src.types import (
    BiasType,
    Data,
    DataJAX,
    DisableBit,
    DynType,
    GainType,
    Model,
    ModelJAX,
    OptionJAX,
)

__all__: list[str] = ['BiasType', 'Data', 'DataJAX', 'DisableBit', 'DynType', 'GainType', 'Model', 'ModelJAX', 'OptionJAX', 'deriv_smooth_vel', 'jax', 'jp']
def deriv_smooth_vel(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[jax.Array]:
    """
    Analytical derivative of smooth forces w.r.t. velocities.
    """
