"""
Core non-smooth constraint functions.
"""
from __future__ import annotations

import dataclasses
import typing

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from jax import numpy as jp
from mujoco.mjx._src import collision_driver, math, support
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src.types import (
    ConeType,
    ConstraintType,
    Contact,
    Data,
    DataJAX,
    DisableBit,
    EqType,
    JointType,
    Model,
    ModelJAX,
    ObjType,
    OptionJAX,
)

__all__: list[str] = ['ConeType', 'ConstraintType', 'Contact', 'Data', 'DataJAX', 'DisableBit', 'EqType', 'JointType', 'Model', 'ModelJAX', 'ObjType', 'OptionJAX', 'PyTreeNode', 'collision_driver', 'counts', 'jax', 'jp', 'make_constraint', 'make_efc_address', 'make_efc_type', 'math', 'mujoco', 'np', 'support']
class _Efc(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Support data for creating constraint matrices.
    """
    __dataclass_fields__: typing.ClassVar[dict]  # value = {'J': Field(name='J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pos_aref': Field(name='pos_aref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pos_imp': Field(name='pos_imp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'invweight': Field(name='invweight',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solref': Field(name='solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solimp': Field(name='solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'margin': Field(name='margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'frictionloss': Field(name='frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ('J', 'pos_aref', 'pos_imp', 'invweight', 'solref', 'solimp', 'margin', 'frictionloss')
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
    def __init__(self, J: jax.Array, pos_aref: jax.Array, pos_imp: jax.Array, invweight: jax.Array, solref: jax.Array, solimp: jax.Array, margin: jax.Array, frictionloss: jax.Array) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
    def __setattr__(self, name, value):
        ...
def _efc_contact_elliptic(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, condim: int) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for frictional elliptic contacts.
    """
def _efc_contact_frictionless(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for frictionless contacts.
    """
def _efc_contact_pyramidal(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, condim: int) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for frictional pyramidal contacts.
    """
def _efc_equality_connect(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for connect equality constraints.
    """
def _efc_equality_joint(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for joint equality constraints.
    """
def _efc_equality_tendon(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for tendon equality constraints.
    """
def _efc_equality_weld(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for weld equality constraints.
    """
def _efc_friction(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for dof frictionloss.
    """
def _efc_limit_ball(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for ball joint limits.
    """
def _efc_limit_slide_hinge(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for slide and hinge joint limits.
    """
def _efc_limit_tendon(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Optional[mujoco.mjx._src.constraint._Efc]:
    """
    Calculates constraint rows for tendon limits.
    """
def _kbi(m: mujoco.mjx._src.types.Model, solref: jax.Array, solimp: jax.Array, pos: jax.Array) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Calculates stiffness, damping, and impedance of a constraint.
    """
def _row(j: jax.Array, *args) -> _Efc:
    """
    Creates an efc row, ensuring args all have same row count.
    """
def counts(efc_type: numpy.ndarray) -> typing.Tuple[int, int, int, int]:
    """
    Returns equality, friction, limit, and contact constraint counts.
    """
def make_constraint(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Creates constraint jacobians and other supporting data.
    """
def make_efc_address(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], dim: numpy.ndarray, efc_type: numpy.ndarray) -> numpy.ndarray:
    """
    Returns efc_address that maps contacts to constraint row address.
    """
def make_efc_type(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], dim: typing.Optional[numpy.ndarray] = None) -> numpy.ndarray:
    """
    Returns efc_type that outlines the type of each constraint row.
    """
