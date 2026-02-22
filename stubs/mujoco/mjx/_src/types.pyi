"""
Base types used in MJX.
"""

from __future__ import annotations

import dataclasses
import enum as enum
import typing
import warnings as warnings

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx.warp import types as mjxw_types

__all__: list[str] = [
    "BiasType",
    "CamLightType",
    "ConeType",
    "ConstraintType",
    "Contact",
    "ConvexMesh",
    "Data",
    "DataC",
    "DataJAX",
    "DisableBit",
    "DynType",
    "EnableBit",
    "EqType",
    "GainType",
    "GeomType",
    "Impl",
    "IntegratorType",
    "JacobianType",
    "JointType",
    "Model",
    "ModelC",
    "ModelJAX",
    "ObjType",
    "Option",
    "OptionC",
    "OptionJAX",
    "PyTreeNode",
    "SensorType",
    "SolverType",
    "Statistic",
    "StatisticWarp",
    "TrnType",
    "WrapType",
    "enum",
    "jax",
    "mjxw_types",
    "mujoco",
    "np",
    "warnings",
]

class BiasType(enum.IntEnum):
    """
    Type of actuator bias.

    Members:
      NONE: no bias
      AFFINE: const + kp*length + kv*velocity
      MUSCLE: muscle passive force computed by muscle_bias
    """

    AFFINE: typing.ClassVar[BiasType]  # value = <BiasType.AFFINE: 1>
    MUSCLE: typing.ClassVar[BiasType]  # value = <BiasType.MUSCLE: 2>
    NONE: typing.ClassVar[BiasType]  # value = <BiasType.NONE: 0>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class CamLightType(enum.IntEnum):
    """
    Type of camera light.

    Members:
      FIXED: pos and rot fixed in body
      TRACK: pos tracks body, rot fixed in global
      TRACKCOM: pos tracks subtree com, rot fixed in body
      TARGETBODY: pos fixed in body, rot tracks target body
      TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
    """

    FIXED: typing.ClassVar[CamLightType]  # value = <CamLightType.FIXED: 0>
    TARGETBODY: typing.ClassVar[CamLightType]  # value = <CamLightType.TARGETBODY: 3>
    TARGETBODYCOM: typing.ClassVar[
        CamLightType
    ]  # value = <CamLightType.TARGETBODYCOM: 4>
    TRACK: typing.ClassVar[CamLightType]  # value = <CamLightType.TRACK: 1>
    TRACKCOM: typing.ClassVar[CamLightType]  # value = <CamLightType.TRACKCOM: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class ConeType(enum.IntEnum):
    """
    Type of friction cone.

    Members:
      PYRAMIDAL: pyramidal
      ELLIPTIC: elliptic
    """

    ELLIPTIC: typing.ClassVar[ConeType]  # value = <ConeType.ELLIPTIC: 1>
    PYRAMIDAL: typing.ClassVar[ConeType]  # value = <ConeType.PYRAMIDAL: 0>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class ConstraintType(enum.IntEnum):
    """
    Type of constraint.

    Members:
      EQUALITY: equality constraint
      LIMIT_JOINT: joint limit
      LIMIT_TENDON: tendon limit
      CONTACT_FRICTIONLESS: frictionless contact
      CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
    """

    CONTACT_ELLIPTIC: typing.ClassVar[
        ConstraintType
    ]  # value = <ConstraintType.CONTACT_ELLIPTIC: 7>
    CONTACT_FRICTIONLESS: typing.ClassVar[
        ConstraintType
    ]  # value = <ConstraintType.CONTACT_FRICTIONLESS: 5>
    CONTACT_PYRAMIDAL: typing.ClassVar[
        ConstraintType
    ]  # value = <ConstraintType.CONTACT_PYRAMIDAL: 6>
    EQUALITY: typing.ClassVar[ConstraintType]  # value = <ConstraintType.EQUALITY: 0>
    FRICTION_DOF: typing.ClassVar[
        ConstraintType
    ]  # value = <ConstraintType.FRICTION_DOF: 1>
    FRICTION_TENDON: typing.ClassVar[
        ConstraintType
    ]  # value = <ConstraintType.FRICTION_TENDON: 2>
    LIMIT_JOINT: typing.ClassVar[
        ConstraintType
    ]  # value = <ConstraintType.LIMIT_JOINT: 3>
    LIMIT_TENDON: typing.ClassVar[
        ConstraintType
    ]  # value = <ConstraintType.LIMIT_TENDON: 4>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class Contact(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Result of collision detection functions.

    Attributes:
      dist: distance between nearest points; neg: penetration
      pos: position of contact point: midpoint between geoms            (3,)
      frame: normal is in [0-2]                                         (9,)
      includemargin: include if dist<includemargin=margin-gap           (1,)
      friction: tangent1, 2, spin, roll1, 2                             (5,)
      solref: constraint solver reference, normal direction             (mjNREF,)
      solreffriction: constraint solver reference, friction directions  (mjNREF,)
      solimp: constraint solver impedance                               (mjNIMP,)
      dim: contact space dimensionality: 1, 3, 4, or 6
      geom1: id of geom 1; deprecated, use geom[0]
      geom2: id of geom 2; deprecated, use geom[1]
      geom: geom ids                                                    (2,)
      efc_address: address in efc; -1: not included
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'dist': Field(name='dist',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pos': Field(name='pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'frame': Field(name='frame',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'includemargin': Field(name='includemargin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'friction': Field(name='friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solref': Field(name='solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solreffriction': Field(name='solreffriction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solimp': Field(name='solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dim': Field(name='dim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom1': Field(name='geom1',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom2': Field(name='geom2',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom': Field(name='geom',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_address': Field(name='efc_address',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "dist",
        "pos",
        "frame",
        "includemargin",
        "friction",
        "solref",
        "solreffriction",
        "solimp",
        "dim",
        "geom1",
        "geom2",
        "geom",
        "efc_address",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        dist: jax.Array,
        pos: jax.Array,
        frame: jax.Array,
        includemargin: jax.Array,
        friction: jax.Array,
        solref: jax.Array,
        solreffriction: jax.Array,
        solimp: jax.Array,
        dim: numpy.ndarray,
        geom1: jax.Array,
        geom2: jax.Array,
        geom: jax.Array,
        efc_address: numpy.ndarray,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class ConvexMesh(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Geom properties for convex meshes.

    Members:
      vert: vertices of the convex mesh
      face: faces of the convex mesh
      face_normal: normal vectors for the faces
      edge: edge indexes for all edges in the convex mesh
      edge_face_normal: indexes for face normals adjacent to edges in `edge`
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'vert': Field(name='vert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'face': Field(name='face',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'face_normal': Field(name='face_normal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'edge': Field(name='edge',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'edge_face_normal': Field(name='edge_face_normal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "vert",
        "face",
        "face_normal",
        "edge",
        "edge_face_normal",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        vert: jax.Array,
        face: jax.Array,
        face_normal: jax.Array,
        edge: jax.Array,
        edge_face_normal: jax.Array,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class Data(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Dynamic state that updates each step.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'time': Field(name='time',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qpos': Field(name='qpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qvel': Field(name='qvel',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'act': Field(name='act',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qacc_warmstart': Field(name='qacc_warmstart',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'plugin_state': Field(name='plugin_state',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ctrl': Field(name='ctrl',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_applied': Field(name='qfrc_applied',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xfrc_applied': Field(name='xfrc_applied',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_active': Field(name='eq_active',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mocap_pos': Field(name='mocap_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mocap_quat': Field(name='mocap_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qacc': Field(name='qacc',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'act_dot': Field(name='act_dot',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'userdata': Field(name='userdata',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensordata': Field(name='sensordata',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xpos': Field(name='xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xquat': Field(name='xquat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xmat': Field(name='xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xipos': Field(name='xipos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ximat': Field(name='ximat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xanchor': Field(name='xanchor',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'xaxis': Field(name='xaxis',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_length': Field(name='ten_length',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_xpos': Field(name='geom_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_xmat': Field(name='geom_xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_xpos': Field(name='site_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_xmat': Field(name='site_xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_xpos': Field(name='cam_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_xmat': Field(name='cam_xmat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subtree_com': Field(name='subtree_com',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cvel': Field(name='cvel',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_bias': Field(name='qfrc_bias',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_gravcomp': Field(name='qfrc_gravcomp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_fluid': Field(name='qfrc_fluid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_passive': Field(name='qfrc_passive',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_actuator': Field(name='qfrc_actuator',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_force': Field(name='actuator_force',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_smooth': Field(name='qfrc_smooth',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qacc_smooth': Field(name='qacc_smooth',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_constraint': Field(name='qfrc_constraint',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_inverse': Field(name='qfrc_inverse',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), '_impl': Field(name='_impl',type=typing.Union[mujoco.mjx._src.types.DataC, mujoco.mjx._src.types.DataJAX, mujoco.mjx.warp.types.DataWarp],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "time",
        "qpos",
        "qvel",
        "act",
        "qacc_warmstart",
        "plugin_state",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "eq_active",
        "mocap_pos",
        "mocap_quat",
        "qacc",
        "act_dot",
        "userdata",
        "sensordata",
        "xpos",
        "xquat",
        "xmat",
        "xipos",
        "ximat",
        "xanchor",
        "xaxis",
        "ten_length",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "cam_xpos",
        "cam_xmat",
        "subtree_com",
        "cvel",
        "qfrc_bias",
        "qfrc_gravcomp",
        "qfrc_fluid",
        "qfrc_passive",
        "qfrc_actuator",
        "actuator_force",
        "qfrc_smooth",
        "qacc_smooth",
        "qfrc_constraint",
        "qfrc_inverse",
        "_impl",
    )
    def replace(self, **changes):
        """
        Return a new object replacing specified fields with new values.
        """
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __getattr__(self, name: str): ...
    def __getitem__(self, key): ...
    def __hash__(self): ...
    def __init__(
        self,
        time: jax.Array,
        qpos: jax.Array,
        qvel: jax.Array,
        act: jax.Array,
        qacc_warmstart: jax.Array,
        plugin_state: jax.Array,
        ctrl: jax.Array,
        qfrc_applied: jax.Array,
        xfrc_applied: jax.Array,
        eq_active: jax.Array,
        mocap_pos: jax.Array,
        mocap_quat: jax.Array,
        qacc: jax.Array,
        act_dot: jax.Array,
        userdata: jax.Array,
        sensordata: jax.Array,
        xpos: jax.Array,
        xquat: jax.Array,
        xmat: jax.Array,
        xipos: jax.Array,
        ximat: jax.Array,
        xanchor: jax.Array,
        xaxis: jax.Array,
        ten_length: jax.Array,
        geom_xpos: jax.Array,
        geom_xmat: jax.Array,
        site_xpos: jax.Array,
        site_xmat: jax.Array,
        cam_xpos: jax.Array,
        cam_xmat: jax.Array,
        subtree_com: jax.Array,
        cvel: jax.Array,
        qfrc_bias: jax.Array,
        qfrc_gravcomp: jax.Array,
        qfrc_fluid: jax.Array,
        qfrc_passive: jax.Array,
        qfrc_actuator: jax.Array,
        actuator_force: jax.Array,
        qfrc_smooth: jax.Array,
        qacc_smooth: jax.Array,
        qfrc_constraint: jax.Array,
        qfrc_inverse: jax.Array,
        _impl: typing.Union[
            mujoco.mjx._src.types.DataC,
            mujoco.mjx._src.types.DataJAX,
            mujoco.mjx.warp.types.DataWarp,
        ],
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...
    def bind(
        self,
        model: Model,
        obj: typing.Union[
            mujoco._specs.MjsBody,
            mujoco._specs.MjsFrame,
            mujoco._specs.MjsGeom,
            mujoco._specs.MjsJoint,
            mujoco._specs.MjsLight,
            mujoco._specs.MjsMaterial,
            mujoco._specs.MjsSite,
            mujoco._specs.MjsMesh,
            mujoco._specs.MjsSkin,
            mujoco._specs.MjsTexture,
            mujoco._specs.MjsText,
            mujoco._specs.MjsTuple,
            mujoco._specs.MjsCamera,
            mujoco._specs.MjsFlex,
            mujoco._specs.MjsHField,
            mujoco._specs.MjsKey,
            mujoco._specs.MjsNumeric,
            mujoco._specs.MjsPair,
            mujoco._specs.MjsExclude,
            mujoco._specs.MjsEquality,
            mujoco._specs.MjsTendon,
            mujoco._specs.MjsSensor,
            mujoco._specs.MjsActuator,
            mujoco._specs.MjsPlugin,
            collections.abc.Iterable[
                typing.Union[
                    mujoco._specs.MjsBody,
                    mujoco._specs.MjsFrame,
                    mujoco._specs.MjsGeom,
                    mujoco._specs.MjsJoint,
                    mujoco._specs.MjsLight,
                    mujoco._specs.MjsMaterial,
                    mujoco._specs.MjsSite,
                    mujoco._specs.MjsMesh,
                    mujoco._specs.MjsSkin,
                    mujoco._specs.MjsTexture,
                    mujoco._specs.MjsText,
                    mujoco._specs.MjsTuple,
                    mujoco._specs.MjsCamera,
                    mujoco._specs.MjsFlex,
                    mujoco._specs.MjsHField,
                    mujoco._specs.MjsKey,
                    mujoco._specs.MjsNumeric,
                    mujoco._specs.MjsPair,
                    mujoco._specs.MjsExclude,
                    mujoco._specs.MjsEquality,
                    mujoco._specs.MjsTendon,
                    mujoco._specs.MjsSensor,
                    mujoco._specs.MjsActuator,
                    mujoco._specs.MjsPlugin,
                ]
            ],
        ],
    ) -> mujoco.mjx._src.support.BindData:
        """
        Bind a Mujoco spec to an MJX Data.
        """
    @property
    def impl(self) -> Impl: ...

class DataC(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    C-specific data.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'ne': Field(name='ne',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nf': Field(name='nf',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nl': Field(name='nl',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nefc': Field(name='nefc',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ncon': Field(name='ncon',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solver_niter': Field(name='solver_niter',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cdof': Field(name='cdof',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cinert': Field(name='cinert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_xpos': Field(name='light_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_xdir': Field(name='light_xdir',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexvert_xpos': Field(name='flexvert_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexelem_aabb': Field(name='flexelem_aabb',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_J_rownnz': Field(name='flexedge_J_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_J_rowadr': Field(name='flexedge_J_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_J_colind': Field(name='flexedge_J_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_J': Field(name='flexedge_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_length': Field(name='flexedge_length',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_wrapadr': Field(name='ten_wrapadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_wrapnum': Field(name='ten_wrapnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_J_rownnz': Field(name='ten_J_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_J_rowadr': Field(name='ten_J_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_J_colind': Field(name='ten_J_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_J': Field(name='ten_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_obj': Field(name='wrap_obj',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_xpos': Field(name='wrap_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_length': Field(name='actuator_length',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'moment_rownnz': Field(name='moment_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'moment_rowadr': Field(name='moment_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'moment_colind': Field(name='moment_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_moment': Field(name='actuator_moment',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'crb': Field(name='crb',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qM': Field(name='qM',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'M': Field(name='M',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qLD': Field(name='qLD',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qLDiagInv': Field(name='qLDiagInv',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'bvh_aabb_dyn': Field(name='bvh_aabb_dyn',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'bvh_active': Field(name='bvh_active',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_velocity': Field(name='flexedge_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_velocity': Field(name='ten_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_velocity': Field(name='actuator_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cdof_dot': Field(name='cdof_dot',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'plugin_data': Field(name='plugin_data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qH': Field(name='qH',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qHDiagInv': Field(name='qHDiagInv',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qDeriv': Field(name='qDeriv',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qLU': Field(name='qLU',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_spring': Field(name='qfrc_spring',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_damper': Field(name='qfrc_damper',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cacc': Field(name='cacc',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cfrc_int': Field(name='cfrc_int',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cfrc_ext': Field(name='cfrc_ext',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subtree_linvel': Field(name='subtree_linvel',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subtree_angmom': Field(name='subtree_angmom',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'contact': Field(name='contact',type=<class 'mujoco.mjx._src.types.Contact'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_type': Field(name='efc_type',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_J': Field(name='efc_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_pos': Field(name='efc_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_margin': Field(name='efc_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_frictionloss': Field(name='efc_frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_D': Field(name='efc_D',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_aref': Field(name='efc_aref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_force': Field(name='efc_force',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "ne",
        "nf",
        "nl",
        "nefc",
        "ncon",
        "solver_niter",
        "cdof",
        "cinert",
        "light_xpos",
        "light_xdir",
        "flexvert_xpos",
        "flexelem_aabb",
        "flexedge_J_rownnz",
        "flexedge_J_rowadr",
        "flexedge_J_colind",
        "flexedge_J",
        "flexedge_length",
        "ten_wrapadr",
        "ten_wrapnum",
        "ten_J_rownnz",
        "ten_J_rowadr",
        "ten_J_colind",
        "ten_J",
        "wrap_obj",
        "wrap_xpos",
        "actuator_length",
        "moment_rownnz",
        "moment_rowadr",
        "moment_colind",
        "actuator_moment",
        "crb",
        "qM",
        "M",
        "qLD",
        "qLDiagInv",
        "bvh_aabb_dyn",
        "bvh_active",
        "flexedge_velocity",
        "ten_velocity",
        "actuator_velocity",
        "cdof_dot",
        "plugin_data",
        "qH",
        "qHDiagInv",
        "qDeriv",
        "qLU",
        "qfrc_spring",
        "qfrc_damper",
        "cacc",
        "cfrc_int",
        "cfrc_ext",
        "subtree_linvel",
        "subtree_angmom",
        "contact",
        "efc_type",
        "efc_J",
        "efc_pos",
        "efc_margin",
        "efc_frictionloss",
        "efc_D",
        "efc_aref",
        "efc_force",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        ne: int,
        nf: int,
        nl: int,
        nefc: int,
        ncon: int,
        solver_niter: jax.Array,
        cdof: jax.Array,
        cinert: jax.Array,
        light_xpos: jax.Array,
        light_xdir: jax.Array,
        flexvert_xpos: jax.Array,
        flexelem_aabb: jax.Array,
        flexedge_J_rownnz: jax.Array,
        flexedge_J_rowadr: jax.Array,
        flexedge_J_colind: jax.Array,
        flexedge_J: jax.Array,
        flexedge_length: jax.Array,
        ten_wrapadr: jax.Array,
        ten_wrapnum: jax.Array,
        ten_J_rownnz: jax.Array,
        ten_J_rowadr: jax.Array,
        ten_J_colind: jax.Array,
        ten_J: jax.Array,
        wrap_obj: jax.Array,
        wrap_xpos: jax.Array,
        actuator_length: jax.Array,
        moment_rownnz: jax.Array,
        moment_rowadr: jax.Array,
        moment_colind: jax.Array,
        actuator_moment: jax.Array,
        crb: jax.Array,
        qM: jax.Array,
        M: jax.Array,
        qLD: jax.Array,
        qLDiagInv: jax.Array,
        bvh_aabb_dyn: jax.Array,
        bvh_active: jax.Array,
        flexedge_velocity: jax.Array,
        ten_velocity: jax.Array,
        actuator_velocity: jax.Array,
        cdof_dot: jax.Array,
        plugin_data: jax.Array,
        qH: jax.Array,
        qHDiagInv: jax.Array,
        qDeriv: jax.Array,
        qLU: jax.Array,
        qfrc_spring: jax.Array,
        qfrc_damper: jax.Array,
        cacc: jax.Array,
        cfrc_int: jax.Array,
        cfrc_ext: jax.Array,
        subtree_linvel: jax.Array,
        subtree_angmom: jax.Array,
        contact: Contact,
        efc_type: jax.Array,
        efc_J: jax.Array,
        efc_pos: jax.Array,
        efc_margin: jax.Array,
        efc_frictionloss: jax.Array,
        efc_D: jax.Array,
        efc_aref: jax.Array,
        efc_force: jax.Array,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class DataJAX(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    JAX-specific data.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'ne': Field(name='ne',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nf': Field(name='nf',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nl': Field(name='nl',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nefc': Field(name='nefc',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ncon': Field(name='ncon',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solver_niter': Field(name='solver_niter',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cdof': Field(name='cdof',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cinert': Field(name='cinert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_wrapadr': Field(name='ten_wrapadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_wrapnum': Field(name='ten_wrapnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_J': Field(name='ten_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_obj': Field(name='wrap_obj',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_xpos': Field(name='wrap_xpos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_length': Field(name='actuator_length',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_moment': Field(name='actuator_moment',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'crb': Field(name='crb',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qM': Field(name='qM',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'M': Field(name='M',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qLD': Field(name='qLD',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qLDiagInv': Field(name='qLDiagInv',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ten_velocity': Field(name='ten_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_velocity': Field(name='actuator_velocity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cdof_dot': Field(name='cdof_dot',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cacc': Field(name='cacc',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cfrc_int': Field(name='cfrc_int',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cfrc_ext': Field(name='cfrc_ext',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subtree_linvel': Field(name='subtree_linvel',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'subtree_angmom': Field(name='subtree_angmom',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'contact': Field(name='contact',type=<class 'mujoco.mjx._src.types.Contact'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_type': Field(name='efc_type',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_J': Field(name='efc_J',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_pos': Field(name='efc_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_margin': Field(name='efc_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_frictionloss': Field(name='efc_frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_D': Field(name='efc_D',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_aref': Field(name='efc_aref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_force': Field(name='efc_force',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "ne",
        "nf",
        "nl",
        "nefc",
        "ncon",
        "solver_niter",
        "cdof",
        "cinert",
        "ten_wrapadr",
        "ten_wrapnum",
        "ten_J",
        "wrap_obj",
        "wrap_xpos",
        "actuator_length",
        "actuator_moment",
        "crb",
        "qM",
        "M",
        "qLD",
        "qLDiagInv",
        "ten_velocity",
        "actuator_velocity",
        "cdof_dot",
        "cacc",
        "cfrc_int",
        "cfrc_ext",
        "subtree_linvel",
        "subtree_angmom",
        "contact",
        "efc_type",
        "efc_J",
        "efc_pos",
        "efc_margin",
        "efc_frictionloss",
        "efc_D",
        "efc_aref",
        "efc_force",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        ne: int,
        nf: int,
        nl: int,
        nefc: int,
        ncon: int,
        solver_niter: jax.Array,
        cdof: jax.Array,
        cinert: jax.Array,
        ten_wrapadr: jax.Array,
        ten_wrapnum: jax.Array,
        ten_J: jax.Array,
        wrap_obj: jax.Array,
        wrap_xpos: jax.Array,
        actuator_length: jax.Array,
        actuator_moment: jax.Array,
        crb: jax.Array,
        qM: jax.Array,
        M: jax.Array,
        qLD: jax.Array,
        qLDiagInv: jax.Array,
        ten_velocity: jax.Array,
        actuator_velocity: jax.Array,
        cdof_dot: jax.Array,
        cacc: jax.Array,
        cfrc_int: jax.Array,
        cfrc_ext: jax.Array,
        subtree_linvel: jax.Array,
        subtree_angmom: jax.Array,
        contact: Contact,
        efc_type: jax.Array,
        efc_J: jax.Array,
        efc_pos: jax.Array,
        efc_margin: jax.Array,
        efc_frictionloss: jax.Array,
        efc_D: jax.Array,
        efc_aref: jax.Array,
        efc_force: jax.Array,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class DisableBit(enum.IntFlag):
    """
    Disable default feature bitflags.

    Members:
      CONSTRAINT:   entire constraint solver
      EQUALITY:     equality constraints
      FRICTIONLOSS: joint and tendon frictionloss constraints
      LIMIT:        joint and tendon limit constraints
      CONTACT:      contact constraints
      SPRING:       passive spring forces
      DAMPER:       passive damper forces
      GRAVITY:      gravitational forces
      CLAMPCTRL:    clamp control to specified range
      WARMSTART:    warmstart constraint solver
      ACTUATION:    apply actuation forces
      REFSAFE:      integrator safety: make ref[0]>=2*timestep
      SENSOR:       sensors
    """

    ACTUATION: typing.ClassVar[DisableBit]  # value = <DisableBit.ACTUATION: 2048>
    CLAMPCTRL: typing.ClassVar[DisableBit]  # value = <DisableBit.CLAMPCTRL: 256>
    CONSTRAINT: typing.ClassVar[DisableBit]  # value = <DisableBit.CONSTRAINT: 1>
    CONTACT: typing.ClassVar[DisableBit]  # value = <DisableBit.CONTACT: 16>
    DAMPER: typing.ClassVar[DisableBit]  # value = <DisableBit.DAMPER: 64>
    EQUALITY: typing.ClassVar[DisableBit]  # value = <DisableBit.EQUALITY: 2>
    EULERDAMP: typing.ClassVar[DisableBit]  # value = <DisableBit.EULERDAMP: 32768>
    FILTERPARENT: typing.ClassVar[DisableBit]  # value = <DisableBit.FILTERPARENT: 1024>
    FRICTIONLOSS: typing.ClassVar[DisableBit]  # value = <DisableBit.FRICTIONLOSS: 4>
    GRAVITY: typing.ClassVar[DisableBit]  # value = <DisableBit.GRAVITY: 128>
    LIMIT: typing.ClassVar[DisableBit]  # value = <DisableBit.LIMIT: 8>
    REFSAFE: typing.ClassVar[DisableBit]  # value = <DisableBit.REFSAFE: 4096>
    SENSOR: typing.ClassVar[DisableBit]  # value = <DisableBit.SENSOR: 8192>
    SPRING: typing.ClassVar[DisableBit]  # value = <DisableBit.SPRING: 32>
    WARMSTART: typing.ClassVar[DisableBit]  # value = <DisableBit.WARMSTART: 512>
    @classmethod
    def __new__(cls, value): ...
    def __and__(self, other): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self): ...
    def __or__(self, other): ...
    def __rand__(self, other): ...
    def __ror__(self, other): ...
    def __rxor__(self, other): ...
    def __xor__(self, other): ...

class DynType(enum.IntEnum):
    """
    Type of actuator dynamics.

    Members:
      NONE: no internal dynamics; ctrl specifies force
      INTEGRATOR: integrator: da/dt = u
      FILTER: linear filter: da/dt = (u-a) / tau
      FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
      MUSCLE: piece-wise linear filter with two time constants
    """

    FILTER: typing.ClassVar[DynType]  # value = <DynType.FILTER: 2>
    FILTEREXACT: typing.ClassVar[DynType]  # value = <DynType.FILTEREXACT: 3>
    INTEGRATOR: typing.ClassVar[DynType]  # value = <DynType.INTEGRATOR: 1>
    MUSCLE: typing.ClassVar[DynType]  # value = <DynType.MUSCLE: 4>
    NONE: typing.ClassVar[DynType]  # value = <DynType.NONE: 0>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class EnableBit(enum.IntFlag):
    """
    Enable optional feature bitflags.

    Members:
      INVDISCRETE: discrete-time inverse dynamics
    """

    INVDISCRETE: typing.ClassVar[EnableBit]  # value = <EnableBit.INVDISCRETE: 8>
    MULTICCD: typing.ClassVar[EnableBit]  # value = <EnableBit.MULTICCD: 16>
    @classmethod
    def __new__(cls, value): ...
    def __and__(self, other): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self): ...
    def __or__(self, other): ...
    def __rand__(self, other): ...
    def __ror__(self, other): ...
    def __rxor__(self, other): ...
    def __xor__(self, other): ...

class EqType(enum.IntEnum):
    """
    Type of equality constraint.

    Members:
      CONNECT: connect two bodies at a point (ball joint)
      WELD: fix relative position and orientation of two bodies
      JOINT: couple the values of two scalar joints with cubic
      TENDON: couple the lengths of two tendons with cubic
    """

    CONNECT: typing.ClassVar[EqType]  # value = <EqType.CONNECT: 0>
    JOINT: typing.ClassVar[EqType]  # value = <EqType.JOINT: 2>
    TENDON: typing.ClassVar[EqType]  # value = <EqType.TENDON: 3>
    WELD: typing.ClassVar[EqType]  # value = <EqType.WELD: 1>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class GainType(enum.IntEnum):
    """
    Type of actuator gain.

    Members:
      FIXED: fixed gain
      AFFINE: const + kp*length + kv*velocity
      MUSCLE: muscle FLV curve computed by muscle_gain
    """

    AFFINE: typing.ClassVar[GainType]  # value = <GainType.AFFINE: 1>
    FIXED: typing.ClassVar[GainType]  # value = <GainType.FIXED: 0>
    MUSCLE: typing.ClassVar[GainType]  # value = <GainType.MUSCLE: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class GeomType(enum.IntEnum):
    """
    Type of geometry.

    Members:
      PLANE: plane
      HFIELD: height field
      SPHERE: sphere
      CAPSULE: capsule
      ELLIPSOID: ellipsoid
      CYLINDER: cylinder
      BOX: box
      MESH: mesh
      SDF: signed distance field
    """

    BOX: typing.ClassVar[GeomType]  # value = <GeomType.BOX: 6>
    CAPSULE: typing.ClassVar[GeomType]  # value = <GeomType.CAPSULE: 3>
    CYLINDER: typing.ClassVar[GeomType]  # value = <GeomType.CYLINDER: 5>
    ELLIPSOID: typing.ClassVar[GeomType]  # value = <GeomType.ELLIPSOID: 4>
    HFIELD: typing.ClassVar[GeomType]  # value = <GeomType.HFIELD: 1>
    MESH: typing.ClassVar[GeomType]  # value = <GeomType.MESH: 7>
    PLANE: typing.ClassVar[GeomType]  # value = <GeomType.PLANE: 0>
    SPHERE: typing.ClassVar[GeomType]  # value = <GeomType.SPHERE: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class Impl(enum.Enum):
    """
    Implementation to use.
    """

    C: typing.ClassVar[Impl]  # value = <Impl.C: 'c'>
    JAX: typing.ClassVar[Impl]  # value = <Impl.JAX: 'jax'>
    WARP: typing.ClassVar[Impl]  # value = <Impl.WARP: 'warp'>

class IntegratorType(enum.IntEnum):
    """
    Integrator mode.

    Members:
      EULER: semi-implicit Euler
      RK4: 4th-order Runge Kutta
      IMPLICITFAST: implicit in velocity, no rne derivative
    """

    EULER: typing.ClassVar[IntegratorType]  # value = <IntegratorType.EULER: 0>
    IMPLICITFAST: typing.ClassVar[
        IntegratorType
    ]  # value = <IntegratorType.IMPLICITFAST: 3>
    RK4: typing.ClassVar[IntegratorType]  # value = <IntegratorType.RK4: 1>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class JacobianType(enum.IntEnum):
    """
    Type of constraint Jacobian.

    Members:
      DENSE: dense
      SPARSE: sparse
      AUTO: sparse if nv>60 and device is TPU, dense otherwise
    """

    AUTO: typing.ClassVar[JacobianType]  # value = <JacobianType.AUTO: 2>
    DENSE: typing.ClassVar[JacobianType]  # value = <JacobianType.DENSE: 0>
    SPARSE: typing.ClassVar[JacobianType]  # value = <JacobianType.SPARSE: 1>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class JointType(enum.IntEnum):
    """
    Type of degree of freedom.

    Members:
      FREE:  global position and orientation (quat)       (7,)
      BALL:  orientation (quat) relative to parent        (4,)
      SLIDE: sliding distance along body-fixed axis       (1,)
      HINGE: rotation angle (rad) around body-fixed axis  (1,)
    """

    BALL: typing.ClassVar[JointType]  # value = <JointType.BALL: 1>
    FREE: typing.ClassVar[JointType]  # value = <JointType.FREE: 0>
    HINGE: typing.ClassVar[JointType]  # value = <JointType.HINGE: 3>
    SLIDE: typing.ClassVar[JointType]  # value = <JointType.SLIDE: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class Model(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Static model of the scene that remains unchanged with each physics step.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'nq': Field(name='nq',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nv': Field(name='nv',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nu': Field(name='nu',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'na': Field(name='na',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nbody': Field(name='nbody',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'njnt': Field(name='njnt',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ngeom': Field(name='ngeom',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nsite': Field(name='nsite',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ncam': Field(name='ncam',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nlight': Field(name='nlight',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmesh': Field(name='nmesh',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshvert': Field(name='nmeshvert',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshnormal': Field(name='nmeshnormal',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshtexcoord': Field(name='nmeshtexcoord',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshface': Field(name='nmeshface',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshgraph': Field(name='nmeshgraph',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshpoly': Field(name='nmeshpoly',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshpolyvert': Field(name='nmeshpolyvert',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmeshpolymap': Field(name='nmeshpolymap',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nhfield': Field(name='nhfield',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nhfielddata': Field(name='nhfielddata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntex': Field(name='ntex',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntexdata': Field(name='ntexdata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmat': Field(name='nmat',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'npair': Field(name='npair',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nexclude': Field(name='nexclude',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'neq': Field(name='neq',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntendon': Field(name='ntendon',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nwrap': Field(name='nwrap',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nsensor': Field(name='nsensor',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nnumeric': Field(name='nnumeric',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntuple': Field(name='ntuple',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nkey': Field(name='nkey',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nmocap': Field(name='nmocap',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nM': Field(name='nM',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nB': Field(name='nB',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nC': Field(name='nC',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nD': Field(name='nD',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nJmom': Field(name='nJmom',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ngravcomp': Field(name='ngravcomp',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nuserdata': Field(name='nuserdata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nsensordata': Field(name='nsensordata',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'npluginstate': Field(name='npluginstate',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'opt': Field(name='opt',type=<class 'mujoco.mjx._src.types.Option'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'stat': Field(name='stat',type=typing.Union[mujoco.mjx._src.types.Statistic, mujoco.mjx._src.types.StatisticWarp],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qpos0': Field(name='qpos0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qpos_spring': Field(name='qpos_spring',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_parentid': Field(name='body_parentid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_mocapid': Field(name='body_mocapid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_rootid': Field(name='body_rootid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_weldid': Field(name='body_weldid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_jntnum': Field(name='body_jntnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_jntadr': Field(name='body_jntadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_sameframe': Field(name='body_sameframe',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_dofnum': Field(name='body_dofnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_dofadr': Field(name='body_dofadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_treeid': Field(name='body_treeid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_geomnum': Field(name='body_geomnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_geomadr': Field(name='body_geomadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_simple': Field(name='body_simple',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_pos': Field(name='body_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_quat': Field(name='body_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_ipos': Field(name='body_ipos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_iquat': Field(name='body_iquat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_mass': Field(name='body_mass',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_subtreemass': Field(name='body_subtreemass',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_inertia': Field(name='body_inertia',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_gravcomp': Field(name='body_gravcomp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_margin': Field(name='body_margin',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_contype': Field(name='body_contype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_conaffinity': Field(name='body_conaffinity',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_invweight0': Field(name='body_invweight0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_type': Field(name='jnt_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_qposadr': Field(name='jnt_qposadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_dofadr': Field(name='jnt_dofadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_bodyid': Field(name='jnt_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_limited': Field(name='jnt_limited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_actfrclimited': Field(name='jnt_actfrclimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_actgravcomp': Field(name='jnt_actgravcomp',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_solref': Field(name='jnt_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_solimp': Field(name='jnt_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_pos': Field(name='jnt_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_axis': Field(name='jnt_axis',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_stiffness': Field(name='jnt_stiffness',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_range': Field(name='jnt_range',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_actfrcrange': Field(name='jnt_actfrcrange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jnt_margin': Field(name='jnt_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_bodyid': Field(name='dof_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_jntid': Field(name='dof_jntid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_parentid': Field(name='dof_parentid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_treeid': Field(name='dof_treeid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_Madr': Field(name='dof_Madr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_simplenum': Field(name='dof_simplenum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_solref': Field(name='dof_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_solimp': Field(name='dof_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_frictionloss': Field(name='dof_frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_armature': Field(name='dof_armature',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_damping': Field(name='dof_damping',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_invweight0': Field(name='dof_invweight0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dof_M0': Field(name='dof_M0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_type': Field(name='geom_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_contype': Field(name='geom_contype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_conaffinity': Field(name='geom_conaffinity',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_condim': Field(name='geom_condim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_bodyid': Field(name='geom_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_sameframe': Field(name='geom_sameframe',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_dataid': Field(name='geom_dataid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_group': Field(name='geom_group',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_matid': Field(name='geom_matid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_priority': Field(name='geom_priority',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_solmix': Field(name='geom_solmix',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_solref': Field(name='geom_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_solimp': Field(name='geom_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_size': Field(name='geom_size',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_aabb': Field(name='geom_aabb',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_rbound': Field(name='geom_rbound',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_pos': Field(name='geom_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_quat': Field(name='geom_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_friction': Field(name='geom_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_margin': Field(name='geom_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_gap': Field(name='geom_gap',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_fluid': Field(name='geom_fluid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_rgba': Field(name='geom_rgba',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_type': Field(name='site_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_bodyid': Field(name='site_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_sameframe': Field(name='site_sameframe',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_size': Field(name='site_size',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_pos': Field(name='site_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'site_quat': Field(name='site_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_mode': Field(name='cam_mode',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_bodyid': Field(name='cam_bodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_targetbodyid': Field(name='cam_targetbodyid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_pos': Field(name='cam_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_quat': Field(name='cam_quat',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_poscom0': Field(name='cam_poscom0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_pos0': Field(name='cam_pos0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_mat0': Field(name='cam_mat0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_fovy': Field(name='cam_fovy',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_resolution': Field(name='cam_resolution',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_sensorsize': Field(name='cam_sensorsize',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cam_intrinsic': Field(name='cam_intrinsic',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_mode': Field(name='light_mode',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_type': Field(name='light_type',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_castshadow': Field(name='light_castshadow',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_pos': Field(name='light_pos',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_dir': Field(name='light_dir',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_poscom0': Field(name='light_poscom0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_pos0': Field(name='light_pos0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_dir0': Field(name='light_dir0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_cutoff': Field(name='light_cutoff',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_vertadr': Field(name='mesh_vertadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_vertnum': Field(name='mesh_vertnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_faceadr': Field(name='mesh_faceadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_bvhadr': Field(name='mesh_bvhadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_bvhnum': Field(name='mesh_bvhnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_octadr': Field(name='mesh_octadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_octnum': Field(name='mesh_octnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_normaladr': Field(name='mesh_normaladr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_normalnum': Field(name='mesh_normalnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_graphadr': Field(name='mesh_graphadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_vert': Field(name='mesh_vert',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_normal': Field(name='mesh_normal',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_face': Field(name='mesh_face',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_graph': Field(name='mesh_graph',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_pos': Field(name='mesh_pos',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_quat': Field(name='mesh_quat',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_texcoordadr': Field(name='mesh_texcoordadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_texcoordnum': Field(name='mesh_texcoordnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_texcoord': Field(name='mesh_texcoord',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_size': Field(name='hfield_size',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_nrow': Field(name='hfield_nrow',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_ncol': Field(name='hfield_ncol',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_adr': Field(name='hfield_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hfield_data': Field(name='hfield_data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_type': Field(name='tex_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_height': Field(name='tex_height',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_width': Field(name='tex_width',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_nchannel': Field(name='tex_nchannel',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_adr': Field(name='tex_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tex_data': Field(name='tex_data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mat_rgba': Field(name='mat_rgba',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mat_texid': Field(name='mat_texid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_dim': Field(name='pair_dim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_geom1': Field(name='pair_geom1',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_geom2': Field(name='pair_geom2',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_signature': Field(name='pair_signature',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_solref': Field(name='pair_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_solreffriction': Field(name='pair_solreffriction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_solimp': Field(name='pair_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_margin': Field(name='pair_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_gap': Field(name='pair_gap',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'pair_friction': Field(name='pair_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'exclude_signature': Field(name='exclude_signature',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_type': Field(name='eq_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_obj1id': Field(name='eq_obj1id',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_obj2id': Field(name='eq_obj2id',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_objtype': Field(name='eq_objtype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_active0': Field(name='eq_active0',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_solref': Field(name='eq_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_solimp': Field(name='eq_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'eq_data': Field(name='eq_data',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_adr': Field(name='tendon_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_num': Field(name='tendon_num',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_limited': Field(name='tendon_limited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_actfrclimited': Field(name='tendon_actfrclimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solref_lim': Field(name='tendon_solref_lim',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solimp_lim': Field(name='tendon_solimp_lim',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solref_fri': Field(name='tendon_solref_fri',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_solimp_fri': Field(name='tendon_solimp_fri',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_range': Field(name='tendon_range',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_actfrcrange': Field(name='tendon_actfrcrange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_margin': Field(name='tendon_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_stiffness': Field(name='tendon_stiffness',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_damping': Field(name='tendon_damping',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_armature': Field(name='tendon_armature',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_frictionloss': Field(name='tendon_frictionloss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_lengthspring': Field(name='tendon_lengthspring',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_length0': Field(name='tendon_length0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_invweight0': Field(name='tendon_invweight0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_type': Field(name='wrap_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_objid': Field(name='wrap_objid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_prm': Field(name='wrap_prm',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_trntype': Field(name='actuator_trntype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_dyntype': Field(name='actuator_dyntype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_gaintype': Field(name='actuator_gaintype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_biastype': Field(name='actuator_biastype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_trnid': Field(name='actuator_trnid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actadr': Field(name='actuator_actadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actnum': Field(name='actuator_actnum',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_group': Field(name='actuator_group',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_ctrllimited': Field(name='actuator_ctrllimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_forcelimited': Field(name='actuator_forcelimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actlimited': Field(name='actuator_actlimited',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_dynprm': Field(name='actuator_dynprm',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_gainprm': Field(name='actuator_gainprm',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_biasprm': Field(name='actuator_biasprm',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actearly': Field(name='actuator_actearly',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_ctrlrange': Field(name='actuator_ctrlrange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_forcerange': Field(name='actuator_forcerange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_actrange': Field(name='actuator_actrange',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_gear': Field(name='actuator_gear',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_cranklength': Field(name='actuator_cranklength',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_acc0': Field(name='actuator_acc0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_lengthrange': Field(name='actuator_lengthrange',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_type': Field(name='sensor_type',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_datatype': Field(name='sensor_datatype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_needstage': Field(name='sensor_needstage',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_objtype': Field(name='sensor_objtype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_objid': Field(name='sensor_objid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_reftype': Field(name='sensor_reftype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_refid': Field(name='sensor_refid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_intprm': Field(name='sensor_intprm',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_dim': Field(name='sensor_dim',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_adr': Field(name='sensor_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_cutoff': Field(name='sensor_cutoff',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'numeric_adr': Field(name='numeric_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'numeric_data': Field(name='numeric_data',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_adr': Field(name='tuple_adr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_size': Field(name='tuple_size',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_objtype': Field(name='tuple_objtype',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_objid': Field(name='tuple_objid',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tuple_objprm': Field(name='tuple_objprm',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_time': Field(name='key_time',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_qpos': Field(name='key_qpos',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_qvel': Field(name='key_qvel',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_act': Field(name='key_act',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_mpos': Field(name='key_mpos',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_mquat': Field(name='key_mquat',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'key_ctrl': Field(name='key_ctrl',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_bodyadr': Field(name='name_bodyadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_jntadr': Field(name='name_jntadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_geomadr': Field(name='name_geomadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_siteadr': Field(name='name_siteadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_camadr': Field(name='name_camadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_meshadr': Field(name='name_meshadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_hfieldadr': Field(name='name_hfieldadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_pairadr': Field(name='name_pairadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_eqadr': Field(name='name_eqadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_tendonadr': Field(name='name_tendonadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_actuatoradr': Field(name='name_actuatoradr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_sensoradr': Field(name='name_sensoradr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_numericadr': Field(name='name_numericadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_tupleadr': Field(name='name_tupleadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'name_keyadr': Field(name='name_keyadr',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'names': Field(name='names',type=<class 'bytes'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'signature': Field(name='signature',type=<class 'numpy.uint64'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), '_sizes': Field(name='_sizes',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), '_impl': Field(name='_impl',type=typing.Union[mujoco.mjx._src.types.ModelC, mujoco.mjx._src.types.ModelJAX, mujoco.mjx.warp.types.ModelWarp],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "nq",
        "nv",
        "nu",
        "na",
        "nbody",
        "njnt",
        "ngeom",
        "nsite",
        "ncam",
        "nlight",
        "nmesh",
        "nmeshvert",
        "nmeshnormal",
        "nmeshtexcoord",
        "nmeshface",
        "nmeshgraph",
        "nmeshpoly",
        "nmeshpolyvert",
        "nmeshpolymap",
        "nhfield",
        "nhfielddata",
        "ntex",
        "ntexdata",
        "nmat",
        "npair",
        "nexclude",
        "neq",
        "ntendon",
        "nwrap",
        "nsensor",
        "nnumeric",
        "ntuple",
        "nkey",
        "nmocap",
        "nM",
        "nB",
        "nC",
        "nD",
        "nJmom",
        "ngravcomp",
        "nuserdata",
        "nsensordata",
        "npluginstate",
        "opt",
        "stat",
        "qpos0",
        "qpos_spring",
        "body_parentid",
        "body_mocapid",
        "body_rootid",
        "body_weldid",
        "body_jntnum",
        "body_jntadr",
        "body_sameframe",
        "body_dofnum",
        "body_dofadr",
        "body_treeid",
        "body_geomnum",
        "body_geomadr",
        "body_simple",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_subtreemass",
        "body_inertia",
        "body_gravcomp",
        "body_margin",
        "body_contype",
        "body_conaffinity",
        "body_invweight0",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_bodyid",
        "jnt_limited",
        "jnt_actfrclimited",
        "jnt_actgravcomp",
        "jnt_solref",
        "jnt_solimp",
        "jnt_pos",
        "jnt_axis",
        "jnt_stiffness",
        "jnt_range",
        "jnt_actfrcrange",
        "jnt_margin",
        "dof_bodyid",
        "dof_jntid",
        "dof_parentid",
        "dof_treeid",
        "dof_Madr",
        "dof_simplenum",
        "dof_solref",
        "dof_solimp",
        "dof_frictionloss",
        "dof_armature",
        "dof_damping",
        "dof_invweight0",
        "dof_M0",
        "geom_type",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_bodyid",
        "geom_sameframe",
        "geom_dataid",
        "geom_group",
        "geom_matid",
        "geom_priority",
        "geom_solmix",
        "geom_solref",
        "geom_solimp",
        "geom_size",
        "geom_aabb",
        "geom_rbound",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_margin",
        "geom_gap",
        "geom_fluid",
        "geom_rgba",
        "site_type",
        "site_bodyid",
        "site_sameframe",
        "site_size",
        "site_pos",
        "site_quat",
        "cam_mode",
        "cam_bodyid",
        "cam_targetbodyid",
        "cam_pos",
        "cam_quat",
        "cam_poscom0",
        "cam_pos0",
        "cam_mat0",
        "cam_fovy",
        "cam_resolution",
        "cam_sensorsize",
        "cam_intrinsic",
        "light_mode",
        "light_type",
        "light_castshadow",
        "light_pos",
        "light_dir",
        "light_poscom0",
        "light_pos0",
        "light_dir0",
        "light_cutoff",
        "mesh_vertadr",
        "mesh_vertnum",
        "mesh_faceadr",
        "mesh_bvhadr",
        "mesh_bvhnum",
        "mesh_octadr",
        "mesh_octnum",
        "mesh_normaladr",
        "mesh_normalnum",
        "mesh_graphadr",
        "mesh_vert",
        "mesh_normal",
        "mesh_face",
        "mesh_graph",
        "mesh_pos",
        "mesh_quat",
        "mesh_texcoordadr",
        "mesh_texcoordnum",
        "mesh_texcoord",
        "hfield_size",
        "hfield_nrow",
        "hfield_ncol",
        "hfield_adr",
        "hfield_data",
        "tex_type",
        "tex_height",
        "tex_width",
        "tex_nchannel",
        "tex_adr",
        "tex_data",
        "mat_rgba",
        "mat_texid",
        "pair_dim",
        "pair_geom1",
        "pair_geom2",
        "pair_signature",
        "pair_solref",
        "pair_solreffriction",
        "pair_solimp",
        "pair_margin",
        "pair_gap",
        "pair_friction",
        "exclude_signature",
        "eq_type",
        "eq_obj1id",
        "eq_obj2id",
        "eq_objtype",
        "eq_active0",
        "eq_solref",
        "eq_solimp",
        "eq_data",
        "tendon_adr",
        "tendon_num",
        "tendon_limited",
        "tendon_actfrclimited",
        "tendon_solref_lim",
        "tendon_solimp_lim",
        "tendon_solref_fri",
        "tendon_solimp_fri",
        "tendon_range",
        "tendon_actfrcrange",
        "tendon_margin",
        "tendon_stiffness",
        "tendon_damping",
        "tendon_armature",
        "tendon_frictionloss",
        "tendon_lengthspring",
        "tendon_length0",
        "tendon_invweight0",
        "wrap_type",
        "wrap_objid",
        "wrap_prm",
        "actuator_trntype",
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
        "actuator_trnid",
        "actuator_actadr",
        "actuator_actnum",
        "actuator_group",
        "actuator_ctrllimited",
        "actuator_forcelimited",
        "actuator_actlimited",
        "actuator_dynprm",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_actearly",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_actrange",
        "actuator_gear",
        "actuator_cranklength",
        "actuator_acc0",
        "actuator_lengthrange",
        "sensor_type",
        "sensor_datatype",
        "sensor_needstage",
        "sensor_objtype",
        "sensor_objid",
        "sensor_reftype",
        "sensor_refid",
        "sensor_intprm",
        "sensor_dim",
        "sensor_adr",
        "sensor_cutoff",
        "numeric_adr",
        "numeric_data",
        "tuple_adr",
        "tuple_size",
        "tuple_objtype",
        "tuple_objid",
        "tuple_objprm",
        "key_time",
        "key_qpos",
        "key_qvel",
        "key_act",
        "key_mpos",
        "key_mquat",
        "key_ctrl",
        "name_bodyadr",
        "name_jntadr",
        "name_geomadr",
        "name_siteadr",
        "name_camadr",
        "name_meshadr",
        "name_hfieldadr",
        "name_pairadr",
        "name_eqadr",
        "name_tendonadr",
        "name_actuatoradr",
        "name_sensoradr",
        "name_numericadr",
        "name_tupleadr",
        "name_keyadr",
        "names",
        "signature",
        "_sizes",
        "_impl",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __getattr__(self, name: str): ...
    def __hash__(self): ...
    def __init__(
        self,
        nq: int,
        nv: int,
        nu: int,
        na: int,
        nbody: int,
        njnt: int,
        ngeom: int,
        nsite: int,
        ncam: int,
        nlight: int,
        nmesh: int,
        nmeshvert: int,
        nmeshnormal: int,
        nmeshtexcoord: int,
        nmeshface: int,
        nmeshgraph: int,
        nmeshpoly: int,
        nmeshpolyvert: int,
        nmeshpolymap: int,
        nhfield: int,
        nhfielddata: int,
        ntex: int,
        ntexdata: int,
        nmat: int,
        npair: int,
        nexclude: int,
        neq: int,
        ntendon: int,
        nwrap: int,
        nsensor: int,
        nnumeric: int,
        ntuple: int,
        nkey: int,
        nmocap: int,
        nM: int,
        nB: int,
        nC: int,
        nD: int,
        nJmom: int,
        ngravcomp: int,
        nuserdata: int,
        nsensordata: int,
        npluginstate: int,
        opt: Option,
        stat: typing.Union[
            mujoco.mjx._src.types.Statistic, mujoco.mjx._src.types.StatisticWarp
        ],
        qpos0: jax.Array,
        qpos_spring: jax.Array,
        body_parentid: numpy.ndarray,
        body_mocapid: numpy.ndarray,
        body_rootid: numpy.ndarray,
        body_weldid: numpy.ndarray,
        body_jntnum: numpy.ndarray,
        body_jntadr: numpy.ndarray,
        body_sameframe: numpy.ndarray,
        body_dofnum: numpy.ndarray,
        body_dofadr: numpy.ndarray,
        body_treeid: numpy.ndarray,
        body_geomnum: numpy.ndarray,
        body_geomadr: numpy.ndarray,
        body_simple: numpy.ndarray,
        body_pos: jax.Array,
        body_quat: jax.Array,
        body_ipos: jax.Array,
        body_iquat: jax.Array,
        body_mass: jax.Array,
        body_subtreemass: jax.Array,
        body_inertia: jax.Array,
        body_gravcomp: jax.Array,
        body_margin: numpy.ndarray,
        body_contype: numpy.ndarray,
        body_conaffinity: numpy.ndarray,
        body_invweight0: jax.Array,
        jnt_type: numpy.ndarray,
        jnt_qposadr: numpy.ndarray,
        jnt_dofadr: numpy.ndarray,
        jnt_bodyid: numpy.ndarray,
        jnt_limited: numpy.ndarray,
        jnt_actfrclimited: numpy.ndarray,
        jnt_actgravcomp: numpy.ndarray,
        jnt_solref: jax.Array,
        jnt_solimp: jax.Array,
        jnt_pos: jax.Array,
        jnt_axis: jax.Array,
        jnt_stiffness: jax.Array,
        jnt_range: jax.Array,
        jnt_actfrcrange: jax.Array,
        jnt_margin: jax.Array,
        dof_bodyid: numpy.ndarray,
        dof_jntid: numpy.ndarray,
        dof_parentid: numpy.ndarray,
        dof_treeid: numpy.ndarray,
        dof_Madr: numpy.ndarray,
        dof_simplenum: numpy.ndarray,
        dof_solref: jax.Array,
        dof_solimp: jax.Array,
        dof_frictionloss: jax.Array,
        dof_armature: jax.Array,
        dof_damping: jax.Array,
        dof_invweight0: jax.Array,
        dof_M0: jax.Array,
        geom_type: numpy.ndarray,
        geom_contype: numpy.ndarray,
        geom_conaffinity: numpy.ndarray,
        geom_condim: numpy.ndarray,
        geom_bodyid: numpy.ndarray,
        geom_sameframe: numpy.ndarray,
        geom_dataid: numpy.ndarray,
        geom_group: numpy.ndarray,
        geom_matid: jax.Array,
        geom_priority: numpy.ndarray,
        geom_solmix: jax.Array,
        geom_solref: jax.Array,
        geom_solimp: jax.Array,
        geom_size: jax.Array,
        geom_aabb: numpy.ndarray,
        geom_rbound: jax.Array,
        geom_pos: jax.Array,
        geom_quat: jax.Array,
        geom_friction: jax.Array,
        geom_margin: jax.Array,
        geom_gap: jax.Array,
        geom_fluid: numpy.ndarray,
        geom_rgba: jax.Array,
        site_type: numpy.ndarray,
        site_bodyid: numpy.ndarray,
        site_sameframe: numpy.ndarray,
        site_size: numpy.ndarray,
        site_pos: jax.Array,
        site_quat: jax.Array,
        cam_mode: numpy.ndarray,
        cam_bodyid: numpy.ndarray,
        cam_targetbodyid: numpy.ndarray,
        cam_pos: jax.Array,
        cam_quat: jax.Array,
        cam_poscom0: jax.Array,
        cam_pos0: jax.Array,
        cam_mat0: jax.Array,
        cam_fovy: numpy.ndarray,
        cam_resolution: numpy.ndarray,
        cam_sensorsize: numpy.ndarray,
        cam_intrinsic: numpy.ndarray,
        light_mode: numpy.ndarray,
        light_type: jax.Array,
        light_castshadow: jax.Array,
        light_pos: jax.Array,
        light_dir: jax.Array,
        light_poscom0: jax.Array,
        light_pos0: jax.Array,
        light_dir0: jax.Array,
        light_cutoff: jax.Array,
        mesh_vertadr: numpy.ndarray,
        mesh_vertnum: numpy.ndarray,
        mesh_faceadr: numpy.ndarray,
        mesh_bvhadr: numpy.ndarray,
        mesh_bvhnum: numpy.ndarray,
        mesh_octadr: numpy.ndarray,
        mesh_octnum: numpy.ndarray,
        mesh_normaladr: numpy.ndarray,
        mesh_normalnum: numpy.ndarray,
        mesh_graphadr: numpy.ndarray,
        mesh_vert: numpy.ndarray,
        mesh_normal: numpy.ndarray,
        mesh_face: numpy.ndarray,
        mesh_graph: numpy.ndarray,
        mesh_pos: numpy.ndarray,
        mesh_quat: numpy.ndarray,
        mesh_texcoordadr: numpy.ndarray,
        mesh_texcoordnum: numpy.ndarray,
        mesh_texcoord: numpy.ndarray,
        hfield_size: numpy.ndarray,
        hfield_nrow: numpy.ndarray,
        hfield_ncol: numpy.ndarray,
        hfield_adr: numpy.ndarray,
        hfield_data: jax.Array,
        tex_type: numpy.ndarray,
        tex_height: numpy.ndarray,
        tex_width: numpy.ndarray,
        tex_nchannel: numpy.ndarray,
        tex_adr: numpy.ndarray,
        tex_data: jax.Array,
        mat_rgba: jax.Array,
        mat_texid: numpy.ndarray,
        pair_dim: numpy.ndarray,
        pair_geom1: numpy.ndarray,
        pair_geom2: numpy.ndarray,
        pair_signature: numpy.ndarray,
        pair_solref: jax.Array,
        pair_solreffriction: jax.Array,
        pair_solimp: jax.Array,
        pair_margin: jax.Array,
        pair_gap: jax.Array,
        pair_friction: jax.Array,
        exclude_signature: numpy.ndarray,
        eq_type: numpy.ndarray,
        eq_obj1id: numpy.ndarray,
        eq_obj2id: numpy.ndarray,
        eq_objtype: numpy.ndarray,
        eq_active0: numpy.ndarray,
        eq_solref: jax.Array,
        eq_solimp: jax.Array,
        eq_data: jax.Array,
        tendon_adr: numpy.ndarray,
        tendon_num: numpy.ndarray,
        tendon_limited: numpy.ndarray,
        tendon_actfrclimited: numpy.ndarray,
        tendon_solref_lim: jax.Array,
        tendon_solimp_lim: jax.Array,
        tendon_solref_fri: jax.Array,
        tendon_solimp_fri: jax.Array,
        tendon_range: jax.Array,
        tendon_actfrcrange: jax.Array,
        tendon_margin: jax.Array,
        tendon_stiffness: jax.Array,
        tendon_damping: jax.Array,
        tendon_armature: jax.Array,
        tendon_frictionloss: jax.Array,
        tendon_lengthspring: jax.Array,
        tendon_length0: jax.Array,
        tendon_invweight0: jax.Array,
        wrap_type: numpy.ndarray,
        wrap_objid: numpy.ndarray,
        wrap_prm: numpy.ndarray,
        actuator_trntype: numpy.ndarray,
        actuator_dyntype: numpy.ndarray,
        actuator_gaintype: numpy.ndarray,
        actuator_biastype: numpy.ndarray,
        actuator_trnid: numpy.ndarray,
        actuator_actadr: numpy.ndarray,
        actuator_actnum: numpy.ndarray,
        actuator_group: numpy.ndarray,
        actuator_ctrllimited: numpy.ndarray,
        actuator_forcelimited: numpy.ndarray,
        actuator_actlimited: numpy.ndarray,
        actuator_dynprm: jax.Array,
        actuator_gainprm: jax.Array,
        actuator_biasprm: jax.Array,
        actuator_actearly: numpy.ndarray,
        actuator_ctrlrange: jax.Array,
        actuator_forcerange: jax.Array,
        actuator_actrange: jax.Array,
        actuator_gear: jax.Array,
        actuator_cranklength: numpy.ndarray,
        actuator_acc0: jax.Array,
        actuator_lengthrange: numpy.ndarray,
        sensor_type: numpy.ndarray,
        sensor_datatype: numpy.ndarray,
        sensor_needstage: numpy.ndarray,
        sensor_objtype: numpy.ndarray,
        sensor_objid: numpy.ndarray,
        sensor_reftype: numpy.ndarray,
        sensor_refid: numpy.ndarray,
        sensor_intprm: numpy.ndarray,
        sensor_dim: numpy.ndarray,
        sensor_adr: numpy.ndarray,
        sensor_cutoff: numpy.ndarray,
        numeric_adr: numpy.ndarray,
        numeric_data: numpy.ndarray,
        tuple_adr: numpy.ndarray,
        tuple_size: numpy.ndarray,
        tuple_objtype: numpy.ndarray,
        tuple_objid: numpy.ndarray,
        tuple_objprm: numpy.ndarray,
        key_time: numpy.ndarray,
        key_qpos: numpy.ndarray,
        key_qvel: numpy.ndarray,
        key_act: numpy.ndarray,
        key_mpos: numpy.ndarray,
        key_mquat: numpy.ndarray,
        key_ctrl: numpy.ndarray,
        name_bodyadr: numpy.ndarray,
        name_jntadr: numpy.ndarray,
        name_geomadr: numpy.ndarray,
        name_siteadr: numpy.ndarray,
        name_camadr: numpy.ndarray,
        name_meshadr: numpy.ndarray,
        name_hfieldadr: numpy.ndarray,
        name_pairadr: numpy.ndarray,
        name_eqadr: numpy.ndarray,
        name_tendonadr: numpy.ndarray,
        name_actuatoradr: numpy.ndarray,
        name_sensoradr: numpy.ndarray,
        name_numericadr: numpy.ndarray,
        name_tupleadr: numpy.ndarray,
        name_keyadr: numpy.ndarray,
        names: bytes,
        signature: numpy.uint64,
        _sizes: jax.Array,
        _impl: typing.Union[
            mujoco.mjx._src.types.ModelC,
            mujoco.mjx._src.types.ModelJAX,
            mujoco.mjx.warp.types.ModelWarp,
        ],
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...
    def bind(
        self,
        obj: typing.Union[
            mujoco._specs.MjsBody,
            mujoco._specs.MjsFrame,
            mujoco._specs.MjsGeom,
            mujoco._specs.MjsJoint,
            mujoco._specs.MjsLight,
            mujoco._specs.MjsMaterial,
            mujoco._specs.MjsSite,
            mujoco._specs.MjsMesh,
            mujoco._specs.MjsSkin,
            mujoco._specs.MjsTexture,
            mujoco._specs.MjsText,
            mujoco._specs.MjsTuple,
            mujoco._specs.MjsCamera,
            mujoco._specs.MjsFlex,
            mujoco._specs.MjsHField,
            mujoco._specs.MjsKey,
            mujoco._specs.MjsNumeric,
            mujoco._specs.MjsPair,
            mujoco._specs.MjsExclude,
            mujoco._specs.MjsEquality,
            mujoco._specs.MjsTendon,
            mujoco._specs.MjsSensor,
            mujoco._specs.MjsActuator,
            mujoco._specs.MjsPlugin,
            collections.abc.Iterable[
                typing.Union[
                    mujoco._specs.MjsBody,
                    mujoco._specs.MjsFrame,
                    mujoco._specs.MjsGeom,
                    mujoco._specs.MjsJoint,
                    mujoco._specs.MjsLight,
                    mujoco._specs.MjsMaterial,
                    mujoco._specs.MjsSite,
                    mujoco._specs.MjsMesh,
                    mujoco._specs.MjsSkin,
                    mujoco._specs.MjsTexture,
                    mujoco._specs.MjsText,
                    mujoco._specs.MjsTuple,
                    mujoco._specs.MjsCamera,
                    mujoco._specs.MjsFlex,
                    mujoco._specs.MjsHField,
                    mujoco._specs.MjsKey,
                    mujoco._specs.MjsNumeric,
                    mujoco._specs.MjsPair,
                    mujoco._specs.MjsExclude,
                    mujoco._specs.MjsEquality,
                    mujoco._specs.MjsTendon,
                    mujoco._specs.MjsSensor,
                    mujoco._specs.MjsActuator,
                    mujoco._specs.MjsPlugin,
                ]
            ],
        ],
    ) -> mujoco.mjx._src.support.BindModel:
        """
        Bind a Mujoco spec to an MJX Model.
        """
    @property
    def impl(self) -> Impl: ...

class ModelC(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    CPU-specific model data.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'nbvh': Field(name='nbvh',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nbvhstatic': Field(name='nbvhstatic',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nbvhdynamic': Field(name='nbvhdynamic',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflex': Field(name='nflex',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflexvert': Field(name='nflexvert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflexedge': Field(name='nflexedge',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflexelem': Field(name='nflexelem',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflexelemdata': Field(name='nflexelemdata',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflexshelldata': Field(name='nflexshelldata',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflexevpair': Field(name='nflexevpair',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nflextexcoord': Field(name='nflextexcoord',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'nplugin': Field(name='nplugin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ntree': Field(name='ntree',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'narena': Field(name='narena',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_bvhadr': Field(name='body_bvhadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'body_bvhnum': Field(name='body_bvhnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'bvh_child': Field(name='bvh_child',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'bvh_nodeid': Field(name='bvh_nodeid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'bvh_aabb': Field(name='bvh_aabb',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'oct_child': Field(name='oct_child',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'oct_aabb': Field(name='oct_aabb',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'oct_coeff': Field(name='oct_coeff',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_plugin': Field(name='geom_plugin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_bodyid': Field(name='light_bodyid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'light_targetbodyid': Field(name='light_targetbodyid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_contype': Field(name='flex_contype',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_conaffinity': Field(name='flex_conaffinity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_condim': Field(name='flex_condim',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_priority': Field(name='flex_priority',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_solmix': Field(name='flex_solmix',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_solref': Field(name='flex_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_solimp': Field(name='flex_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_friction': Field(name='flex_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_margin': Field(name='flex_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_gap': Field(name='flex_gap',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_internal': Field(name='flex_internal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_selfcollide': Field(name='flex_selfcollide',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_activelayers': Field(name='flex_activelayers',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_passive': Field(name='flex_passive',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_dim': Field(name='flex_dim',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_vertadr': Field(name='flex_vertadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_vertnum': Field(name='flex_vertnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_edgeadr': Field(name='flex_edgeadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_edgenum': Field(name='flex_edgenum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_elemadr': Field(name='flex_elemadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_elemnum': Field(name='flex_elemnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_elemdataadr': Field(name='flex_elemdataadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_evpairadr': Field(name='flex_evpairadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_evpairnum': Field(name='flex_evpairnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_vertbodyid': Field(name='flex_vertbodyid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_edge': Field(name='flex_edge',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_elem': Field(name='flex_elem',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_elemlayer': Field(name='flex_elemlayer',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_evpair': Field(name='flex_evpair',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_vert': Field(name='flex_vert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_length0': Field(name='flexedge_length0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_invweight0': Field(name='flexedge_invweight0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_radius': Field(name='flex_radius',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_edgestiffness': Field(name='flex_edgestiffness',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_edgedamping': Field(name='flex_edgedamping',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_edgeequality': Field(name='flex_edgeequality',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_rigid': Field(name='flex_rigid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flexedge_rigid': Field(name='flexedge_rigid',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_centered': Field(name='flex_centered',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_bvhadr': Field(name='flex_bvhadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'flex_bvhnum': Field(name='flex_bvhnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'actuator_plugin': Field(name='actuator_plugin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sensor_plugin': Field(name='sensor_plugin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'plugin': Field(name='plugin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'plugin_stateadr': Field(name='plugin_stateadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'B_rownnz': Field(name='B_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'B_rowadr': Field(name='B_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'B_colind': Field(name='B_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'M_rownnz': Field(name='M_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'M_rowadr': Field(name='M_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'M_colind': Field(name='M_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mapM2M': Field(name='mapM2M',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'D_rownnz': Field(name='D_rownnz',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'D_rowadr': Field(name='D_rowadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'D_diag': Field(name='D_diag',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'D_colind': Field(name='D_colind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mapM2D': Field(name='mapM2D',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mapD2M': Field(name='mapD2M',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polynum': Field(name='mesh_polynum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polyadr': Field(name='mesh_polyadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polynormal': Field(name='mesh_polynormal',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polyvertadr': Field(name='mesh_polyvertadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polyvertnum': Field(name='mesh_polyvertnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polyvert': Field(name='mesh_polyvert',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polymapadr': Field(name='mesh_polymapadr',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polymapnum': Field(name='mesh_polymapnum',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_polymap': Field(name='mesh_polymap',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "nbvh",
        "nbvhstatic",
        "nbvhdynamic",
        "nflex",
        "nflexvert",
        "nflexedge",
        "nflexelem",
        "nflexelemdata",
        "nflexshelldata",
        "nflexevpair",
        "nflextexcoord",
        "nplugin",
        "ntree",
        "narena",
        "body_bvhadr",
        "body_bvhnum",
        "bvh_child",
        "bvh_nodeid",
        "bvh_aabb",
        "oct_child",
        "oct_aabb",
        "oct_coeff",
        "geom_plugin",
        "light_bodyid",
        "light_targetbodyid",
        "flex_contype",
        "flex_conaffinity",
        "flex_condim",
        "flex_priority",
        "flex_solmix",
        "flex_solref",
        "flex_solimp",
        "flex_friction",
        "flex_margin",
        "flex_gap",
        "flex_internal",
        "flex_selfcollide",
        "flex_activelayers",
        "flex_passive",
        "flex_dim",
        "flex_vertadr",
        "flex_vertnum",
        "flex_edgeadr",
        "flex_edgenum",
        "flex_elemadr",
        "flex_elemnum",
        "flex_elemdataadr",
        "flex_evpairadr",
        "flex_evpairnum",
        "flex_vertbodyid",
        "flex_edge",
        "flex_elem",
        "flex_elemlayer",
        "flex_evpair",
        "flex_vert",
        "flexedge_length0",
        "flexedge_invweight0",
        "flex_radius",
        "flex_edgestiffness",
        "flex_edgedamping",
        "flex_edgeequality",
        "flex_rigid",
        "flexedge_rigid",
        "flex_centered",
        "flex_bvhadr",
        "flex_bvhnum",
        "actuator_plugin",
        "sensor_plugin",
        "plugin",
        "plugin_stateadr",
        "B_rownnz",
        "B_rowadr",
        "B_colind",
        "M_rownnz",
        "M_rowadr",
        "M_colind",
        "mapM2M",
        "D_rownnz",
        "D_rowadr",
        "D_diag",
        "D_colind",
        "mapM2D",
        "mapD2M",
        "mesh_polynum",
        "mesh_polyadr",
        "mesh_polynormal",
        "mesh_polyvertadr",
        "mesh_polyvertnum",
        "mesh_polyvert",
        "mesh_polymapadr",
        "mesh_polymapnum",
        "mesh_polymap",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        nbvh: jax.Array,
        nbvhstatic: jax.Array,
        nbvhdynamic: jax.Array,
        nflex: jax.Array,
        nflexvert: jax.Array,
        nflexedge: jax.Array,
        nflexelem: jax.Array,
        nflexelemdata: jax.Array,
        nflexshelldata: jax.Array,
        nflexevpair: jax.Array,
        nflextexcoord: jax.Array,
        nplugin: jax.Array,
        ntree: jax.Array,
        narena: jax.Array,
        body_bvhadr: jax.Array,
        body_bvhnum: jax.Array,
        bvh_child: jax.Array,
        bvh_nodeid: jax.Array,
        bvh_aabb: jax.Array,
        oct_child: jax.Array,
        oct_aabb: jax.Array,
        oct_coeff: jax.Array,
        geom_plugin: jax.Array,
        light_bodyid: jax.Array,
        light_targetbodyid: jax.Array,
        flex_contype: jax.Array,
        flex_conaffinity: jax.Array,
        flex_condim: jax.Array,
        flex_priority: jax.Array,
        flex_solmix: jax.Array,
        flex_solref: jax.Array,
        flex_solimp: jax.Array,
        flex_friction: jax.Array,
        flex_margin: jax.Array,
        flex_gap: jax.Array,
        flex_internal: jax.Array,
        flex_selfcollide: jax.Array,
        flex_activelayers: jax.Array,
        flex_passive: jax.Array,
        flex_dim: jax.Array,
        flex_vertadr: jax.Array,
        flex_vertnum: jax.Array,
        flex_edgeadr: jax.Array,
        flex_edgenum: jax.Array,
        flex_elemadr: jax.Array,
        flex_elemnum: jax.Array,
        flex_elemdataadr: jax.Array,
        flex_evpairadr: jax.Array,
        flex_evpairnum: jax.Array,
        flex_vertbodyid: jax.Array,
        flex_edge: jax.Array,
        flex_elem: jax.Array,
        flex_elemlayer: jax.Array,
        flex_evpair: jax.Array,
        flex_vert: jax.Array,
        flexedge_length0: jax.Array,
        flexedge_invweight0: jax.Array,
        flex_radius: jax.Array,
        flex_edgestiffness: jax.Array,
        flex_edgedamping: jax.Array,
        flex_edgeequality: jax.Array,
        flex_rigid: jax.Array,
        flexedge_rigid: jax.Array,
        flex_centered: jax.Array,
        flex_bvhadr: jax.Array,
        flex_bvhnum: jax.Array,
        actuator_plugin: jax.Array,
        sensor_plugin: jax.Array,
        plugin: jax.Array,
        plugin_stateadr: jax.Array,
        B_rownnz: jax.Array,
        B_rowadr: jax.Array,
        B_colind: jax.Array,
        M_rownnz: jax.Array,
        M_rowadr: jax.Array,
        M_colind: jax.Array,
        mapM2M: jax.Array,
        D_rownnz: jax.Array,
        D_rowadr: jax.Array,
        D_diag: jax.Array,
        D_colind: jax.Array,
        mapM2D: jax.Array,
        mapD2M: jax.Array,
        mesh_polynum: jax.Array,
        mesh_polyadr: jax.Array,
        mesh_polynormal: jax.Array,
        mesh_polyvertadr: jax.Array,
        mesh_polyvertnum: jax.Array,
        mesh_polyvert: jax.Array,
        mesh_polymapadr: jax.Array,
        mesh_polymapnum: jax.Array,
        mesh_polymap: jax.Array,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class ModelJAX(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    JAX-specific model data.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'dof_hasfrictionloss': Field(name='dof_hasfrictionloss',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'geom_rbound_hfield': Field(name='geom_rbound_hfield',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'mesh_convex': Field(name='mesh_convex',type=typing.Tuple[mujoco.mjx._src.types.ConvexMesh, ...],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tendon_hasfrictionloss': Field(name='tendon_hasfrictionloss',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_inside_maxiter': Field(name='wrap_inside_maxiter',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_inside_tolerance': Field(name='wrap_inside_tolerance',type=<class 'float'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wrap_inside_z_init': Field(name='wrap_inside_z_init',type=<class 'float'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'is_wrap_inside': Field(name='is_wrap_inside',type=<class 'numpy.ndarray'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "dof_hasfrictionloss",
        "geom_rbound_hfield",
        "mesh_convex",
        "tendon_hasfrictionloss",
        "wrap_inside_maxiter",
        "wrap_inside_tolerance",
        "wrap_inside_z_init",
        "is_wrap_inside",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        dof_hasfrictionloss: numpy.ndarray,
        geom_rbound_hfield: numpy.ndarray,
        mesh_convex: typing.Tuple[mujoco.mjx._src.types.ConvexMesh, ...],
        tendon_hasfrictionloss: numpy.ndarray,
        wrap_inside_maxiter: int,
        wrap_inside_tolerance: float,
        wrap_inside_z_init: float,
        is_wrap_inside: numpy.ndarray,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class ObjType(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Type of object.

    Members:
      UNKNOWN: unknown object type
      BODY: body
      XBODY: body, used to access regular frame instead of i-frame
      GEOM: geom
      SITE: site
      CAMERA: camera
    """

    BODY: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_BODY: 1>
    CAMERA: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_CAMERA: 7>
    GEOM: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_GEOM: 5>
    SITE: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_SITE: 6>
    UNKNOWN: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_UNKNOWN: 0>
    XBODY: typing.ClassVar[mujoco._enums.mjtObj]  # value = <mjtObj.mjOBJ_XBODY: 2>
    __dataclass_fields__: typing.ClassVar[dict] = {}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = tuple()
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(self) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class Option(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Physics options.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'iterations': Field(name='iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ls_iterations': Field(name='ls_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'tolerance': Field(name='tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ls_tolerance': Field(name='ls_tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'impratio': Field(name='impratio',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'gravity': Field(name='gravity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'density': Field(name='density',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'viscosity': Field(name='viscosity',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'magnetic': Field(name='magnetic',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'wind': Field(name='wind',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'jacobian': Field(name='jacobian',type=<enum 'JacobianType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cone': Field(name='cone',type=<enum 'ConeType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'disableflags': Field(name='disableflags',type=<flag 'DisableBit'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'enableflags': Field(name='enableflags',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'integrator': Field(name='integrator',type=<enum 'IntegratorType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solver': Field(name='solver',type=<enum 'SolverType'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'timestep': Field(name='timestep',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), '_impl': Field(name='_impl',type=typing.Union[mujoco.mjx._src.types.OptionJAX, mujoco.mjx._src.types.OptionC, mujoco.mjx.warp.types.OptionWarp],default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "iterations",
        "ls_iterations",
        "tolerance",
        "ls_tolerance",
        "impratio",
        "gravity",
        "density",
        "viscosity",
        "magnetic",
        "wind",
        "jacobian",
        "cone",
        "disableflags",
        "enableflags",
        "integrator",
        "solver",
        "timestep",
        "_impl",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        iterations: int,
        ls_iterations: int,
        tolerance: jax.Array,
        ls_tolerance: jax.Array,
        impratio: jax.Array,
        gravity: jax.Array,
        density: jax.Array,
        viscosity: jax.Array,
        magnetic: jax.Array,
        wind: jax.Array,
        jacobian: JacobianType,
        cone: ConeType,
        disableflags: DisableBit,
        enableflags: int,
        integrator: IntegratorType,
        solver: SolverType,
        timestep: jax.Array,
        _impl: typing.Union[
            mujoco.mjx._src.types.OptionJAX,
            mujoco.mjx._src.types.OptionC,
            mujoco.mjx.warp.types.OptionWarp,
        ],
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class OptionC(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    C-specific option.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'o_margin': Field(name='o_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_solref': Field(name='o_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_solimp': Field(name='o_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_friction': Field(name='o_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'disableactuator': Field(name='disableactuator',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sdf_initpoints': Field(name='sdf_initpoints',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'has_fluid_params': Field(name='has_fluid_params',type=<class 'bool'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'noslip_tolerance': Field(name='noslip_tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ccd_tolerance': Field(name='ccd_tolerance',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'noslip_iterations': Field(name='noslip_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ccd_iterations': Field(name='ccd_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sdf_iterations': Field(name='sdf_iterations',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "o_margin",
        "o_solref",
        "o_solimp",
        "o_friction",
        "disableactuator",
        "sdf_initpoints",
        "has_fluid_params",
        "noslip_tolerance",
        "ccd_tolerance",
        "noslip_iterations",
        "ccd_iterations",
        "sdf_iterations",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        o_margin: jax.Array,
        o_solref: jax.Array,
        o_solimp: jax.Array,
        o_friction: jax.Array,
        disableactuator: int,
        sdf_initpoints: int,
        has_fluid_params: bool,
        noslip_tolerance: jax.Array,
        ccd_tolerance: jax.Array,
        noslip_iterations: int,
        ccd_iterations: int,
        sdf_iterations: int,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class OptionJAX(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    JAX-specific option.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'o_margin': Field(name='o_margin',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_solref': Field(name='o_solref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_solimp': Field(name='o_solimp',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'o_friction': Field(name='o_friction',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'disableactuator': Field(name='disableactuator',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'sdf_initpoints': Field(name='sdf_initpoints',type=<class 'int'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'has_fluid_params': Field(name='has_fluid_params',type=<class 'bool'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "o_margin",
        "o_solref",
        "o_solimp",
        "o_friction",
        "disableactuator",
        "sdf_initpoints",
        "has_fluid_params",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        o_margin: jax.Array,
        o_solref: jax.Array,
        o_solimp: jax.Array,
        o_friction: jax.Array,
        disableactuator: int,
        sdf_initpoints: int,
        has_fluid_params: bool,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class SensorType(enum.IntEnum):
    """
    Type of sensor.

    Members:
      MAGNETOMETER: magnetometer
      CAMPROJECTION: camera projection
      RANGEFINDER: rangefinder
      JOINTPOS: joint position
      TENDONPOS: scalar tendon position
      ACTUATORPOS: actuator position
      BALLQUAT: ball joint orientation
      FRAMEPOS: frame position
      FRAMEXAXIS: frame x-axis
      FRAMEYAXIS: frame y-axis
      FRAMEZAXIS: frame z-axis
      FRAMEQUAT: frame orientation, represented as quaternion
      SUBTREECOM: subtree centor of mass
      CLOCK: simulation time
      VELOCIMETER: 3D linear velocity, in local frame
      GYRO: 3D angular velocity, in local frame
      JOINTVEL: joint velocity
      TENDONVEL: scalar tendon velocity
      ACTUATORVEL: actuator velocity
      BALLANGVEL: ball joint angular velocity
      FRAMELINVEL: 3D linear velocity
      FRAMEANGVEL: 3D angular velocity
      SUBTREELINVEL: subtree linear velocity
      SUBTREEANGMOM: subtree angular momentum
      TOUCH: scalar contact normal forces summed over the sensor zone
      CONTACT: contacts which occurred during the simulation
      ACCELEROMETER: accelerometer
      FORCE: force
      TORQUE: torque
      ACTUATORFRC: scalar actuator force
      JOINTACTFRC: scalar actuator force, measured at the joint
      TENDONACTFRC: scalar actuator force, measured at the tendon
      FRAMELINACC: 3D linear acceleration
      FRAMEANGACC: 3D angular acceleration
    """

    ACCELEROMETER: typing.ClassVar[SensorType]  # value = <SensorType.ACCELEROMETER: 1>
    ACTUATORFRC: typing.ClassVar[SensorType]  # value = <SensorType.ACTUATORFRC: 15>
    ACTUATORPOS: typing.ClassVar[SensorType]  # value = <SensorType.ACTUATORPOS: 13>
    ACTUATORVEL: typing.ClassVar[SensorType]  # value = <SensorType.ACTUATORVEL: 14>
    BALLANGVEL: typing.ClassVar[SensorType]  # value = <SensorType.BALLANGVEL: 19>
    BALLQUAT: typing.ClassVar[SensorType]  # value = <SensorType.BALLQUAT: 18>
    CAMPROJECTION: typing.ClassVar[SensorType]  # value = <SensorType.CAMPROJECTION: 8>
    CLOCK: typing.ClassVar[SensorType]  # value = <SensorType.CLOCK: 45>
    CONTACT: typing.ClassVar[SensorType]  # value = <SensorType.CONTACT: 42>
    FORCE: typing.ClassVar[SensorType]  # value = <SensorType.FORCE: 4>
    FRAMEANGACC: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEANGACC: 34>
    FRAMEANGVEL: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEANGVEL: 32>
    FRAMELINACC: typing.ClassVar[SensorType]  # value = <SensorType.FRAMELINACC: 33>
    FRAMELINVEL: typing.ClassVar[SensorType]  # value = <SensorType.FRAMELINVEL: 31>
    FRAMEPOS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEPOS: 26>
    FRAMEQUAT: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEQUAT: 27>
    FRAMEXAXIS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEXAXIS: 28>
    FRAMEYAXIS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEYAXIS: 29>
    FRAMEZAXIS: typing.ClassVar[SensorType]  # value = <SensorType.FRAMEZAXIS: 30>
    GYRO: typing.ClassVar[SensorType]  # value = <SensorType.GYRO: 3>
    JOINTACTFRC: typing.ClassVar[SensorType]  # value = <SensorType.JOINTACTFRC: 16>
    JOINTPOS: typing.ClassVar[SensorType]  # value = <SensorType.JOINTPOS: 9>
    JOINTVEL: typing.ClassVar[SensorType]  # value = <SensorType.JOINTVEL: 10>
    MAGNETOMETER: typing.ClassVar[SensorType]  # value = <SensorType.MAGNETOMETER: 6>
    RANGEFINDER: typing.ClassVar[SensorType]  # value = <SensorType.RANGEFINDER: 7>
    SUBTREEANGMOM: typing.ClassVar[SensorType]  # value = <SensorType.SUBTREEANGMOM: 37>
    SUBTREECOM: typing.ClassVar[SensorType]  # value = <SensorType.SUBTREECOM: 35>
    SUBTREELINVEL: typing.ClassVar[SensorType]  # value = <SensorType.SUBTREELINVEL: 36>
    TENDONACTFRC: typing.ClassVar[SensorType]  # value = <SensorType.TENDONACTFRC: 17>
    TENDONPOS: typing.ClassVar[SensorType]  # value = <SensorType.TENDONPOS: 11>
    TENDONVEL: typing.ClassVar[SensorType]  # value = <SensorType.TENDONVEL: 12>
    TORQUE: typing.ClassVar[SensorType]  # value = <SensorType.TORQUE: 5>
    TOUCH: typing.ClassVar[SensorType]  # value = <SensorType.TOUCH: 0>
    VELOCIMETER: typing.ClassVar[SensorType]  # value = <SensorType.VELOCIMETER: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class SolverType(enum.IntEnum):
    """
    Constraint solver algorithm.

    Members:
      CG: Conjugate gradient (primal)
      NEWTON: Newton (primal)
    """

    CG: typing.ClassVar[SolverType]  # value = <SolverType.CG: 1>
    NEWTON: typing.ClassVar[SolverType]  # value = <SolverType.NEWTON: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class Statistic(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Model statistics (in qpos0).

    Attributes:
      meaninertia: mean diagonal inertia
      meanmass: mean body mass (not used)
      meansize: mean body size (not used)
      extent: spatial extent (not used)
      center: center of model (not used)
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'meaninertia': Field(name='meaninertia',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'meanmass': Field(name='meanmass',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'meansize': Field(name='meansize',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'extent': Field(name='extent',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'center': Field(name='center',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "meaninertia",
        "meanmass",
        "meansize",
        "extent",
        "center",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        meaninertia: jax.Array,
        meanmass: jax.Array,
        meansize: jax.Array,
        extent: jax.Array,
        center: jax.Array,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class StatisticWarp(mujoco.mjx.warp.types.StatisticWarp, Statistic):
    """
    Warp-specific model statistics.
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'meaninertia': Field(name='meaninertia',type=<class 'float'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'meanmass': Field(name='meanmass',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'meansize': Field(name='meansize',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'extent': Field(name='extent',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'center': Field(name='center',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "meaninertia",
        "meanmass",
        "meansize",
        "extent",
        "center",
    )
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
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        meaninertia: float,
        meanmass: jax.Array,
        meansize: jax.Array,
        extent: jax.Array,
        center: jax.Array,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class TrnType(enum.IntEnum):
    """
    Type of actuator transmission.

    Members:
      JOINT: force on joint
      JOINTINPARENT: force on joint, expressed in parent frame
      TENDON: force on tendon
      SITE: force on site
    """

    JOINT: typing.ClassVar[TrnType]  # value = <TrnType.JOINT: 0>
    JOINTINPARENT: typing.ClassVar[TrnType]  # value = <TrnType.JOINTINPARENT: 1>
    SITE: typing.ClassVar[TrnType]  # value = <TrnType.SITE: 4>
    TENDON: typing.ClassVar[TrnType]  # value = <TrnType.TENDON: 3>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class WrapType(enum.IntEnum):
    """
    Type of tendon wrap object.

    Members:
      JOINT: constant moment arm
      PULLEY: pulley used to split tendon
      SITE: pass through site
      SPHERE: wrap around sphere
      CYLINDER: wrap around (infinite) cylinder
    """

    CYLINDER: typing.ClassVar[WrapType]  # value = <WrapType.CYLINDER: 5>
    JOINT: typing.ClassVar[WrapType]  # value = <WrapType.JOINT: 1>
    PULLEY: typing.ClassVar[WrapType]  # value = <WrapType.PULLEY: 2>
    SITE: typing.ClassVar[WrapType]  # value = <WrapType.SITE: 3>
    SPHERE: typing.ClassVar[WrapType]  # value = <WrapType.SPHERE: 4>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
