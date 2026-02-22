"""
Some useful math functions.
"""

from __future__ import annotations

import jax as jax
import mujoco as mujoco
from jax import numpy as jp

__all__: list[str] = [
    "axis_angle_to_quat",
    "closest_segment_point",
    "closest_segment_point_and_dist",
    "closest_segment_to_segment_points",
    "inert_mul",
    "jax",
    "jp",
    "make_frame",
    "matmul_unroll",
    "motion_cross",
    "motion_cross_force",
    "mujoco",
    "norm",
    "normalize",
    "normalize_with_norm",
    "orthogonals",
    "quat_integrate",
    "quat_inv",
    "quat_mul",
    "quat_mul_axis",
    "quat_sub",
    "quat_to_axis_angle",
    "quat_to_mat",
    "rotate",
    "safe_div",
    "sign",
    "transform_motion",
]

def axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """
    Provides a quaternion that describes rotating around axis by angle.

    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by

    Returns:
      A quaternion that rotates around axis by angle
    """

def closest_segment_point(a: jax.Array, b: jax.Array, pt: jax.Array) -> jax.Array:
    """
    Returns the closest point on the a-b line segment to a point pt.
    """

def closest_segment_point_and_dist(
    a: jax.Array, b: jax.Array, pt: jax.Array
) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Returns closest point on the line segment and the distance squared.
    """

def closest_segment_to_segment_points(
    a0: jax.Array, a1: jax.Array, b0: jax.Array, b1: jax.Array
) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Returns closest points between two line segments.
    """

def inert_mul(i: jax.Array, v: jax.Array) -> jax.Array:
    """
    Multiply inertia by motion, producing force.

    Args:
      i: (10,) inertia (inertia matrix, position, mass)
      v: (6,) spatial motion

    Returns:
      resultant force
    """

def make_frame(a: jax.Array) -> jax.Array:
    """
    Makes a right-handed 3D frame given a direction.
    """

def matmul_unroll(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Calculates a @ b via explicit cell value operations.

    This is faster than XLA matmul for small matrices (e.g. 3x3, 4x4).

    Args:
      a: left hand of matmul operand
      b: right hand of matmul operand

    Returns:
      the matrix product of the inputs.
    """

def motion_cross(u, v):
    """
    Cross product of two motions.

    Args:
      u: (6,) spatial motion
      v: (6,) spatial motion

    Returns:
      resultant spatial motion
    """

def motion_cross_force(v, f):
    """
    Cross product of a motion and force.

    Args:
      v: (6,) spatial motion
      f: (6,) force

    Returns:
      resultant force
    """

def norm(
    x: jax.Array, axis: typing.Union[typing.Tuple[int, ...], int, NoneType] = None
) -> jax.Array:
    """
    Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Avoids a poorly defined gradient for jnp.linal.norm(0) see
    https://github.com/jax-ml/jax/issues/3058 for details
    Args:
      x: A jnp.array
      axis: The axis along which to compute the norm

    Returns:
      Norm of the array x.
    """

def normalize(
    x: jax.Array, axis: typing.Union[typing.Tuple[int, ...], int, NoneType] = None
) -> jax.Array:
    """
    Normalizes an array.

    Args:
      x: A jnp.array
      axis: The axis along which to compute the norm

    Returns:
      normalized array x
    """

def normalize_with_norm(
    x: jax.Array, axis: typing.Union[typing.Tuple[int, ...], int, NoneType] = None
) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Normalizes an array.

    Args:
      x: A jnp.array
      axis: The axis along which to compute the norm

    Returns:
      A tuple of (normalized array x, the norm).
    """

def orthogonals(a: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Returns orthogonal vectors `b` and `c`, given a vector `a`.
    """

def quat_integrate(q: jax.Array, v: jax.Array, dt: jax.Array) -> jax.Array:
    """
    Integrates a quaternion given angular velocity and dt.
    """

def quat_inv(q: jax.Array) -> jax.Array:
    """
    Calculates the inverse of quaternion q.

    Args:
      q: (4,) quaternion [w, x, y, z]

    Returns:
      The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """

def quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
    """
    Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """

def quat_mul_axis(q: jax.Array, axis: jax.Array) -> jax.Array:
    """
    Multiplies a quaternion and an axis.

    Args:
      q: (4,) quaternion (w,x,y,z)
      axis: (3,) axis (x,y,z)

    Returns:
      A quaternion q * axis
    """

def quat_sub(u: jax.Array, v: jax.Array) -> jax.Array:
    """
    Subtracts two quaternions (u - v) as a 3D velocity.
    """

def quat_to_axis_angle(q: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Converts a quaternion into axis and angle.
    """

def quat_to_mat(q: jax.Array) -> jax.Array:
    """
    Converts a quaternion into a 9-dimensional rotation matrix.
    """

def rotate(vec: jax.Array, quat: jax.Array) -> jax.Array:
    """
    Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by quat.
    """

def safe_div(
    num: typing.Union[float, jax.Array], den: typing.Union[float, jax.Array]
) -> typing.Union[float, jax.Array]:
    """
    Safe division for case where denominator is zero.
    """

def sign(x: jax.Array) -> jax.Array:
    """
    Returns the sign of x in the set {-1, 1}.
    """

def transform_motion(vel: jax.Array, offset: jax.Array, rotmat: jax.Array):
    """
    Transform spatial motion.

    Args:
      vel: (6,) spatial motion (3 angular, 3 linear)
      offset: (3,) translation
      rotmat: (3, 3) rotation

    Returns:
      6d spatial velocity
    """
