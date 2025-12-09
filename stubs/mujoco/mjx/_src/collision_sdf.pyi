"""
Collision functions for shapes represented as signed distance functions (SDF).

A signed distance function at a given point in space is the shortest distance to
a surface. This enables to define a geometry implicitly and exactly.

See https://iquilezles.org/articles/distfunctions/ for a list of analytic SDFs.
"""
from __future__ import annotations

import dataclasses
import functools as functools
import typing

import jax as jax
import mujoco.mjx._src.collision_types
import mujoco.mjx._src.dataclasses
from jax import numpy as jp
from mujoco.mjx._src import math
from mujoco.mjx._src.collision_types import GeomInfo
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src.types import Data, DataJAX, Model, ModelJAX

__all__: list[str] = ['Collision', 'Data', 'DataJAX', 'GeomInfo', 'GradientState', 'Model', 'ModelJAX', 'PyTreeNode', 'SDFFn', 'collider', 'cylinder_jvp', 'functools', 'jax', 'jp', 'math']
class GradientState(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    GradientState(dist: jax.Array, x: jax.Array)
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'dist': Field(name='dist',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'x': Field(name='x',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ('dist', 'x')
    @staticmethod
    def replace(obj, **changes):
        """
        Return a new object replacing specified fields with new values.
        
        This is especially useful for frozen classes.  Example usage::
        
          @dataclass(frozen=True)
          class C:
              x: int
              y: int
        
          c = C(1, 2)
          c1 = replace(c, x=3)
          assert c1.x == 3 and c1.y == 2
        """
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, dist: jax.Array, x: jax.Array) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
def _capsule(pos: jax.Array, size: jax.Array):
    ...
def _clearance(d1: typing.Callable[[jax.Array], jax.Array], d2: typing.Callable[[jax.Array], jax.Array]) -> typing.Callable[[jax.Array], jax.Array]:
    ...
def _cylinder_grad(x: jax.Array, size: jax.Array) -> jax.Array:
    """
    Gradient of the cylinder SDF wrt query point and singularities removed.
    """
def _ellipsoid(pos: jax.Array, size: jax.Array) -> jax.Array:
    ...
def _from_to(f: typing.Callable[[jax.Array], jax.Array], from_pos: jax.Array, from_mat: jax.Array, to_pos: jax.Array, to_mat: jax.Array) -> typing.Callable[[jax.Array], jax.Array]:
    ...
def _gradient_descent(objective: typing.Callable[[jax.Array], jax.Array], x: jax.Array, niter: int) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Performs gradient descent with backtracking line search.
    """
def _gradient_step(objective: typing.Callable[[jax.Array], jax.Array], state: GradientState) -> GradientState:
    """
    Performs a step of gradient descent.
    """
def _intersect(d1: typing.Callable[[jax.Array], jax.Array], d2: typing.Callable[[jax.Array], jax.Array]) -> typing.Callable[[jax.Array], jax.Array]:
    ...
def _optim(d1, d2, info1: mujoco.mjx._src.collision_types.GeomInfo, info2: mujoco.mjx._src.collision_types.GeomInfo, x0: jax.Array) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Optimizes the clearance function.
    """
def _plane(pos: jax.Array, size: jax.Array) -> jax.Array:
    ...
def _sphere(pos: jax.Array, size: jax.Array):
    ...
def collider(ncon: int):
    """
    Wraps collision functions for use by collision_driver.
    """
def cylinder_jvp(primals, tangents):
    ...
Collision: typing._GenericAlias  # value = typing.Tuple[jax.Array, jax.Array, jax.Array]
SDFFn: typing._CallableGenericAlias  # value = typing.Callable[[jax.Array], jax.Array]
_cylinder: jax._src.custom_derivatives.custom_jvp  # value = <jax._src.custom_derivatives.custom_jvp object>
