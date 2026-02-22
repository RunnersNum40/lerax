"""
Collision primitives.
"""

from __future__ import annotations

import typing

import jax as jax
from jax import numpy as jp
from mujoco.mjx._src import math
from mujoco.mjx._src.collision_types import GeomInfo
from mujoco.mjx._src.types import Data, Model

__all__: list[str] = [
    "Collision",
    "Data",
    "GeomInfo",
    "Model",
    "collider",
    "jax",
    "jp",
    "math",
]

def _plane_sphere(
    plane_normal: jax.Array,
    plane_pos: jax.Array,
    sphere_pos: jax.Array,
    sphere_radius: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Returns the distance and contact point between a plane and sphere.
    """

def _sphere_sphere(
    pos1: jax.Array, radius1: jax.Array, pos2: jax.Array, radius2: jax.Array
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns the penetration, contact point, and normal between two spheres.
    """

def collider(ncon: int):
    """
    Wraps collision functions for use by collision_driver.
    """

Collision: typing._GenericAlias  # value = typing.Tuple[jax.Array, jax.Array, jax.Array]
