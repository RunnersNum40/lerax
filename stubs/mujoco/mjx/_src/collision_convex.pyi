"""
Convex collisions.
"""

from __future__ import annotations

import functools as functools
import typing

import jax as jax
import mujoco.mjx._src.collision_types
from jax import numpy as jp
from mujoco.mjx._src import math, mesh
from mujoco.mjx._src.collision_types import (
    ConvexInfo,
    FunctionKey,
    GeomInfo,
    HFieldInfo,
)
from mujoco.mjx._src.types import Data, DataJAX, GeomType, Model, ModelJAX

__all__: list[str] = [
    "Collision",
    "ConvexInfo",
    "Data",
    "DataJAX",
    "FunctionKey",
    "GeomInfo",
    "GeomType",
    "HFieldInfo",
    "Model",
    "ModelJAX",
    "collider",
    "functools",
    "jax",
    "jp",
    "math",
    "mesh",
]

def _arcs_intersect(
    a: jax.Array, b: jax.Array, c: jax.Array, d: jax.Array
) -> jax.Array:
    """
    Tests if arcs AB and CD on the unit sphere intersect.
    """

def _box_box(
    b1: mujoco.mjx._src.collision_types.ConvexInfo,
    b2: mujoco.mjx._src.collision_types.ConvexInfo,
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Calculates contacts between two boxes.
    """

def _box_box_impl(
    faces_a: jax.Array,
    faces_b: jax.Array,
    vertices_a: jax.Array,
    vertices_b: jax.Array,
    normals_a: jax.Array,
    normals_b: jax.Array,
    unique_edges_a: jax.Array,
    unique_edges_b: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Runs the Separating Axis Test for two boxes.

    Args:
      faces_a: Faces for hull A.
      faces_b: Faces for hull B.
      vertices_a: Vertices for hull A.
      vertices_b: Vertices for hull B.
      normals_a: Normal vectors for hull A faces.
      normals_b: Normal vectors for hull B faces.
      unique_edges_a: Unique edges for hull A.
      unique_edges_b: Unique edges for hull B.

    Returns:
      tuple of dist, pos, and normal
    """

def _capsule_convex(
    cap: mujoco.mjx._src.collision_types.GeomInfo,
    convex: mujoco.mjx._src.collision_types.ConvexInfo,
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Calculates contacts between a capsule and a convex object.
    """

def _clip(
    clipping_poly: jax.Array,
    subject_poly: jax.Array,
    clipping_normal: jax.Array,
    subject_normal: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Clips a subject polygon against a clipping polygon.

    A parallelized clipping algorithm for convex polygons. The result is a set of
    vertices on the clipped subject polygon in the subject polygon plane.

    Args:
      clipping_poly: the polygon that we use to clip the subject polygon against
      subject_poly: the polygon that gets clipped
      clipping_normal: normal of the clipping polygon
      subject_normal: normal of the subject polygon

    Returns:
      clipped_pts: points on the clipped polygon
      mask: True if a point is in the clipping polygon, False otherwise
    """

def _clip_edge_to_planes(
    edge_p0: jax.Array,
    edge_p1: jax.Array,
    plane_pts: jax.Array,
    plane_normals: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Clips an edge against side planes.

    We return two clipped points, and a mask to include the new edge or not.

    Args:
      edge_p0: the first point on the edge
      edge_p1: the second point on the edge
      plane_pts: side plane points
      plane_normals: side plane normals

    Returns:
      new_ps: new edge points that are clipped against side planes
      mask: a boolean mask, True if an edge point is a valid clipped point and
      False otherwise
    """

def _closest_segment_point_plane(
    a: jax.Array, b: jax.Array, p0: jax.Array, plane_normal: jax.Array
) -> jax.Array:
    """
    Gets the closest point between a line segment and a plane.

    Args:
      a: first line segment point
      b: second line segment point
      p0: point on plane
      plane_normal: plane normal

    Returns:
      closest point between the line segment and the plane
    """

def _convex_convex(
    c1: mujoco.mjx._src.collision_types.ConvexInfo,
    c2: mujoco.mjx._src.collision_types.ConvexInfo,
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Calculates contacts between two convex meshes.
    """

def _create_contact_manifold(
    clipping_poly: jax.Array,
    subject_poly: jax.Array,
    clipping_norm: jax.Array,
    subject_norm: jax.Array,
    sep_axis: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Creates a contact manifold between two convex polygons.

    The polygon faces are expected to have a counter clockwise winding order so
    that clipping plane normals point away from the polygon center.

    Args:
      clipping_poly: the reference polygon to clip the contact against.
      subject_poly: the subject polygon to clip contacts onto.
      clipping_norm: the clipping polygon normal.
      subject_norm: the subject polygon normal.
      sep_axis: the separating axis

    Returns:
      tuple of dist, pos, and normal
    """

def _hfield_collision(
    collider_fn: typing.Callable[
        [
            typing.Union[
                mujoco.mjx._src.collision_types.GeomInfo,
                mujoco.mjx._src.collision_types.ConvexInfo,
            ],
            typing.Union[
                mujoco.mjx._src.collision_types.GeomInfo,
                mujoco.mjx._src.collision_types.ConvexInfo,
            ],
        ],
        typing.Tuple[jax.Array, jax.Array, jax.Array],
    ],
    h: mujoco.mjx._src.collision_types.HFieldInfo,
    obj: typing.Union[
        mujoco.mjx._src.collision_types.GeomInfo,
        mujoco.mjx._src.collision_types.ConvexInfo,
    ],
    obj_rbound: jax.Array,
    subgrid_size: typing.Tuple[int, int],
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Collides an object with prisms in a height field.
    """

def _manifold_points(
    poly: jax.Array, poly_mask: jax.Array, poly_norm: jax.Array
) -> jax.Array:
    """
    Chooses four points on the polygon with approximately maximal area.
    """

def _point_in_front_of_plane(
    plane_pt: jax.Array, plane_normal: jax.Array, pt: jax.Array
) -> jax.Array:
    """
    Checks if a point is strictly in front of a plane.
    """

def _project_poly_onto_plane(
    poly: jax.Array, plane_pt: jax.Array, plane_normal: jax.Array
) -> jax.Array:
    """
    Projects a polygon onto a plane using the plane normal.
    """

def _project_poly_onto_poly_plane(
    poly1: jax.Array, norm1: jax.Array, poly2: jax.Array, norm2: jax.Array
) -> jax.Array:
    """
    Projects poly1 onto the poly2 plane along poly1's normal.
    """

def _project_pt_onto_plane(
    pt: jax.Array, plane_pt: jax.Array, plane_normal: jax.Array
) -> jax.Array:
    """
    Projects a point onto a plane along the plane normal.
    """

def _sat_gaussmap(
    centroid_a: jax.Array,
    faces_a: jax.Array,
    faces_b: jax.Array,
    vertices_a: jax.Array,
    vertices_b: jax.Array,
    normals_a: jax.Array,
    normals_b: jax.Array,
    edges_a: jax.Array,
    edges_b: jax.Array,
    edge_face_normals_a: jax.Array,
    edge_face_normals_b: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Runs the Separating Axis Test for a pair of hulls.

    Runs the separating axis test for all faces. Tests edge separating axes via
    edge intersections on gauss maps for all edge pairs. h/t to Dirk Gregorius
    for the implementation details and gauss map trick.

    Args:
      centroid_a: Centroid of hull A.
      faces_a: Faces for hull A.
      faces_b: Faces for hull B.
      vertices_a: Vertices for hull A.
      vertices_b: Vertices for hull B.
      normals_a: Normal vectors for hull A faces.
      normals_b: Normal vectors for hull B faces.
      edges_a: Edges for hull A.
      edges_b: Edges for hull B.
      edge_face_normals_a: Face normals for edges in hull A.
      edge_face_normals_b: Face normals for edges in hull B.

    Returns:
      tuple of dist, pos, and normal
    """

def _sphere_convex(
    sphere: mujoco.mjx._src.collision_types.GeomInfo,
    convex: mujoco.mjx._src.collision_types.ConvexInfo,
) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Calculates contact between a sphere and a convex mesh.
    """

def collider(ncon: int):
    """
    Wraps collision functions for use by collision_driver.
    """

Collision: typing._GenericAlias  # value = typing.Tuple[jax.Array, jax.Array, jax.Array]
_GeomInfo: typing._UnionGenericAlias  # value = typing.Union[mujoco.mjx._src.collision_types.GeomInfo, mujoco.mjx._src.collision_types.ConvexInfo]
