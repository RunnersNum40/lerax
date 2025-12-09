"""
Functions for ray interesection testing.
"""
from __future__ import annotations

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from jax import numpy as jp
from mujoco.mjx._src import math
from mujoco.mjx._src.types import Data, GeomType, Model

__all__: list[str] = ['Data', 'GeomType', 'Model', 'jax', 'jp', 'math', 'mujoco', 'np', 'ray', 'ray_geom']
def _ray_box(size: jax.Array, pnt: jax.Array, vec: jax.Array) -> jax.Array:
    """
    Returns the distance at which a ray intersects with a box.
    """
def _ray_capsule(size: jax.Array, pnt: jax.Array, vec: jax.Array) -> jax.Array:
    """
    Returns the distance at which a ray intersects with a capsule.
    """
def _ray_ellipsoid(size: jax.Array, pnt: jax.Array, vec: jax.Array) -> jax.Array:
    """
    Returns the distance at which a ray intersects with an ellipsoid.
    """
def _ray_mesh(m: mujoco.mjx._src.types.Model, geom_id: numpy.ndarray, unused_size: jax.Array, pnt: jax.Array, vec: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Returns the best distance and geom_id for ray mesh intersections.
    """
def _ray_plane(size: jax.Array, pnt: jax.Array, vec: jax.Array) -> jax.Array:
    """
    Returns the distance at which a ray intersects with a plane.
    """
def _ray_quad(a: jax.Array, b: jax.Array, c: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Returns two solutions for quadratic: a*x^2 + 2*b*x + c = 0.
    """
def _ray_sphere(size: jax.Array, pnt: jax.Array, vec: jax.Array) -> jax.Array:
    """
    Returns the distance at which a ray intersects with a sphere.
    """
def _ray_triangle(vert: jax.Array, pnt: jax.Array, vec: jax.Array, basis: jax.Array) -> jax.Array:
    """
    Returns the distance at which a ray intersects with a triangle.
    """
def ray(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, pnt: jax.Array, vec: jax.Array, geomgroup: typing.Sequence[int] = tuple(), flg_static: bool = True, bodyexclude: int = -1) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Returns the geom id and distance at which a ray intersects with a geom.
    
    Args:
      m: MJX model
      d: MJX data
      pnt: ray origin point (3,)
      vec: ray direction    (3,)
      geomgroup: group inclusion/exclusion mask, or empty to ignore
      flg_static: if True, allows rays to intersect with static geoms
      bodyexclude: ignore geoms on specified body id
    
    Returns:
      dist: distance from ray origin to geom surface (or -1.0 for no intersection)
      id: id of intersected geom (or -1 for no intersection)
    """
def ray_geom(size: jax.Array, pnt: jax.Array, vec: jax.Array, geomtype: mujoco.mjx._src.types.GeomType) -> jax.Array:
    """
    Returns the distance at which a ray intersects with a primitive geom.
    
    Args:
      size: geom size (1,), (2,), or (3,)
      pnt: ray origin point (3,)
      vec: ray direction    (3,)
      geomtype: type of geom
    
    Returns:
      dist: distance from ray origin to geom surface
    """
_RAY_FUNC: dict  # value = {<GeomType.PLANE: 0>: _ray_plane, <GeomType.SPHERE: 2>: _ray_sphere, <GeomType.CAPSULE: 3>: _ray_capsule, <GeomType.ELLIPSOID: 4>: _ray_ellipsoid, <GeomType.BOX: 6>: _ray_box, <GeomType.MESH: 7>: _ray_mesh}
