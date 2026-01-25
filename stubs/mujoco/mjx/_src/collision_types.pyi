"""
Collision base types.
"""
from __future__ import annotations

import dataclasses as dataclasses
import typing

import jax as jax
import mujoco.mjx._src.dataclasses
import numpy
import numpy as np
from mujoco.mjx._src.dataclasses import PyTreeNode

__all__: list[str] = ['Collision', 'ConvexInfo', 'FunctionKey', 'GeomInfo', 'HFieldInfo', 'PyTreeNode', 'dataclasses', 'jax', 'np']
class ConvexInfo(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Geom properties for convex meshes.
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'pos': Field(name='pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mat': Field(name='mat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'size': Field(name='size',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'vert': Field(name='vert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'face': Field(name='face',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'face_normal': Field(name='face_normal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'edge': Field(name='edge',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'edge_face_normal': Field(name='edge_face_normal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ('pos', 'mat', 'size', 'vert', 'face', 'face_normal', 'edge', 'edge_face_normal')
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
    def __init__(self, pos: jax.Array, mat: jax.Array, size: jax.Array, vert: jax.Array, face: jax.Array, face_normal: jax.Array, edge: jax.Array, edge_face_normal: jax.Array) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class FunctionKey:
    """
    Specifies how geom pairs group into collision_driver's function table.

    Attributes:
      types: geom type pair, which determines the collision function
      data_ids: geom data id pair: mesh id for mesh geoms, otherwise -1. Meshes
        have distinct face/vertex counts, so must occupy distinct entries in the
        collision function table.
      condim: grouping by condim of the colliision ensures that the size of the
        resulting constraint jacobian is determined at compile time.
      subgrid_size: the size determines the hfield subgrid to collide with
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'types': Field(name='types',type=typing.Tuple[int, int],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'data_ids': Field(name='data_ids',type=typing.Tuple[int, int],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'condim': Field(name='condim',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subgrid_size': Field(name='subgrid_size',type=typing.Tuple[int, int],default=(-1, -1),default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ('types', 'data_ids', 'condim', 'subgrid_size')
    subgrid_size: typing.ClassVar[tuple] = (-1, -1)
    def __delattr__(self, name):
        ...
    def __eq__(self, other):
        ...
    def __hash__(self):
        ...
    def __init__(self, types: typing.Tuple[int, int], data_ids: typing.Tuple[int, int], condim: int, subgrid_size: typing.Tuple[int, int] = (-1, -1)) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class GeomInfo(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Geom properties for primitive shapes.
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'pos': Field(name='pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mat': Field(name='mat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'size': Field(name='size',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ('pos', 'mat', 'size')
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
    def __init__(self, pos: jax.Array, mat: jax.Array, size: jax.Array) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
class HFieldInfo(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Geom properties for height fields.
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'pos': Field(name='pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mat': Field(name='mat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'size': Field(name='size',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nrow': Field(name='nrow',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ncol': Field(name='ncol',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'data': Field(name='data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ('pos', 'mat', 'size', 'nrow', 'ncol', 'data')
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
    def __init__(self, pos: jax.Array, mat: jax.Array, size: numpy.ndarray, nrow: int, ncol: int, data: jax.Array) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
Collision: typing._GenericAlias  # value = typing.Tuple[jax.Array, jax.Array, jax.Array]
