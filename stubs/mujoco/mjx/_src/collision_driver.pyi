"""
Runs collision checking for all geoms in a Model.

To do this, collision_driver builds a collision function table, and then runs
the collision functions serially on the parameters in the table.

For example, if a Model has three geoms:

geom   |   type
---------------
1      | sphere
2      | capsule
3      | sphere

collision_driver organizes it into these functions and runs them:

function       | geom pair
--------------------------
sphere_sphere  | (1, 3)
sphere_capsule | (1, 2), (2, 3)


Besides collision function, function tables are keyed on mesh id and condim,
in order to guarantee static shapes for contacts and jacobians.
"""
from __future__ import annotations

import itertools as itertools
import os as os

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from jax import numpy as jp
from mujoco.mjx._src import support
from mujoco.mjx._src.collision_types import FunctionKey
from mujoco.mjx._src.types import (
    Contact,
    Data,
    DataJAX,
    DisableBit,
    GeomType,
    Model,
    ModelJAX,
    OptionJAX,
)

__all__: list[str] = ['Contact', 'Data', 'DataJAX', 'DisableBit', 'FunctionKey', 'GeomType', 'Model', 'ModelJAX', 'OptionJAX', 'collision', 'geom_pairs', 'has_collision_fn', 'itertools', 'jax', 'jp', 'make_condim', 'mujoco', 'np', 'os', 'support']
def _contact_groups(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Dict[mujoco.mjx._src.collision_types.FunctionKey, mujoco.mjx._src.types.Contact]:
    """
    Returns contact groups to check for collisions.
    
    Contacts are grouped the same way as _geom_groups.  Only one contact is
    emitted per geom pair, even if the collision function emits multiple contacts.
    
    Args:
      m: MJX model
      d: MJX data
    
    Returns:
      a dict where the key is the grouping and value is a Contact
    """
def _geom_groups(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel]) -> typing.Dict[mujoco.mjx._src.collision_types.FunctionKey, typing.List[typing.Tuple[int, int, int]]]:
    """
    Returns geom pairs to check for collision grouped by collision function.
    
    The grouping consists of:
      - The collision function to run, which is determined by geom types
      - For mesh geoms, convex functions are run for each distinct mesh in the
        model, because the convex functions expect static mesh size. If a sphere
        collides with a cube and a tetrahedron, sphere_convex is called twice.
      - The condim of the collision. This ensures that the size of the resulting
        constraint jacobian is determined at compile time.
    
    Args:
      m: a MuJoCo or MJX model
    
    Returns:
      a dict with grouping key and values geom1, geom2, pair index
    """
def _numeric(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], name: str) -> int:
    ...
def collision(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Collides geometries.
    """
def geom_pairs(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel]) -> typing.Iterator[typing.Tuple[int, int, int]]:
    """
    Yields geom pairs to check for collisions.
    
    Args:
      m: a MuJoCo or MJX model
    
    Yields:
      geom1, geom2, and pair index if defined in <pair> (else -1)
    """
def has_collision_fn(t1: mujoco.mjx._src.types.GeomType, t2: mujoco.mjx._src.types.GeomType) -> bool:
    """
    Returns True if a collision function exists for a pair of geom types.
    """
def make_condim(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel]) -> numpy.ndarray:
    """
    Returns the dims of the contacts for a Model.
    """
_COLLISION_FUNC: dict  # value = {(<GeomType.PLANE: 0>, <GeomType.SPHERE: 2>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdfe20>, (<GeomType.PLANE: 0>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdff60>, (<GeomType.PLANE: 0>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecde7a0>, (<GeomType.PLANE: 0>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece80e0>, (<GeomType.PLANE: 0>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece8220>, (<GeomType.PLANE: 0>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecde7a0>, (<GeomType.HFIELD: 1>, <GeomType.SPHERE: 2>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdf740>, (<GeomType.HFIELD: 1>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdf880>, (<GeomType.HFIELD: 1>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdf9c0>, (<GeomType.HFIELD: 1>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdf9c0>, (<GeomType.SPHERE: 2>, <GeomType.SPHERE: 2>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece8400>, (<GeomType.SPHERE: 2>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece8540>, (<GeomType.SPHERE: 2>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece9b20>, (<GeomType.SPHERE: 2>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece99e0>, (<GeomType.SPHERE: 2>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecde980>, (<GeomType.SPHERE: 2>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecde980>, (<GeomType.CAPSULE: 3>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece8680>, (<GeomType.CAPSULE: 3>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdeb60>, (<GeomType.CAPSULE: 3>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece9c60>, (<GeomType.CAPSULE: 3>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece9da0>, (<GeomType.CAPSULE: 3>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdeb60>, (<GeomType.ELLIPSOID: 4>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ece9ee0>, (<GeomType.ELLIPSOID: 4>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecea020>, (<GeomType.CYLINDER: 5>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecea160>, (<GeomType.BOX: 6>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdf420>, (<GeomType.BOX: 6>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdf560>, (<GeomType.MESH: 7>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x7fdf3ecdf560>}
_GEOM_NO_BROADPHASE: set  # value = {<GeomType.PLANE: 0>, <GeomType.HFIELD: 1>}
