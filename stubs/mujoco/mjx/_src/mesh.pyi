"""
Mesh processing.
"""
from __future__ import annotations

import collections as collections
import itertools as itertools
import warnings as warnings

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
import trimesh as trimesh
from jax import numpy as jp
from mujoco.mjx._src import math
from mujoco.mjx._src.collision_types import ConvexInfo, GeomInfo, HFieldInfo
from mujoco.mjx._src.types import ConvexMesh, Model
from scipy import spatial

__all__: list[str] = ['ConvexInfo', 'ConvexMesh', 'GeomInfo', 'HFieldInfo', 'Model', 'box', 'collections', 'convex', 'hfield', 'hfield_prism', 'itertools', 'jax', 'jp', 'math', 'mujoco', 'np', 'spatial', 'trimesh', 'warnings']
def _convex_hull_2d(points: numpy.ndarray, normal: numpy.ndarray) -> numpy.ndarray:
    """
    Calculates the convex hull for a set of points on a plane.
    """
def _get_edge_normals(face: numpy.ndarray, face_norm: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Returns face edges and face edge normals.
    """
def _get_face_norm(vert: numpy.ndarray, face: numpy.ndarray) -> numpy.ndarray:
    """
    Calculates face normals given vertices and face indexes.
    """
def _merge_coplanar(m: typing.Union[mujoco._structs.MjModel, mujoco.mjx._src.types.Model], tm: trimesh.base.Trimesh, meshid: int) -> numpy.ndarray:
    """
    Merges coplanar facets.
    """
def box(info: mujoco.mjx._src.collision_types.GeomInfo) -> mujoco.mjx._src.collision_types.ConvexInfo:
    """
    Creates a box with rectangular faces.
    """
def convex(m: typing.Union[mujoco._structs.MjModel, mujoco.mjx._src.types.Model], data_id: int) -> mujoco.mjx._src.types.ConvexMesh:
    """
    Processes a mesh for use in convex collision algorithms.

    Args:
      m: an MJX model
      data_id: the mesh id to process

    Returns:
      a convex mesh
    """
def hfield(m: typing.Union[mujoco._structs.MjModel, mujoco.mjx._src.types.Model], data_id: int) -> mujoco.mjx._src.collision_types.HFieldInfo:
    ...
def hfield_prism(vert: jax.Array) -> mujoco.mjx._src.collision_types.ConvexInfo:
    """
    Builds a hfield prism.
    """
_MAX_HULL_FACE_VERTICES: int = 20
