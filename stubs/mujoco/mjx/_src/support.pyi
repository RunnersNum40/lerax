"""
Engine support functions.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from jax import numpy as jp
from mujoco.introspect import mjxmacro
from mujoco.mjx._src import math, scan
from mujoco.mjx._src.types import ConeType, Data, JacobianType, JointType, Model

__all__: list[str] = ['BindData', 'BindModel', 'ConeType', 'Data', 'Iterable', 'JacobianType', 'JointType', 'Model', 'Sequence', 'apply_ft', 'contact_force', 'contact_force_dim', 'full_m', 'id2name', 'is_sparse', 'jac', 'jac_dot', 'jax', 'jp', 'local_to_global', 'make_m', 'math', 'mjxmacro', 'mujoco', 'mul_m', 'muscle_bias', 'muscle_dynamics', 'muscle_dynamics_timescale', 'muscle_gain', 'muscle_gain_length', 'name2id', 'np', 'scan', 'wrap', 'wrap_circle', 'wrap_inside', 'xfrc_accumulate']
class BindData:
    """
    Class holding the requested MJX Data and spec id for binding a spec to Data.
    """
    def _BindData__getname(self, name: str):
        """
        Get the name of the attribute and check if the type is correct.
        """
    def __getattr__(self, name: str):
        ...
    def __init__(self, data: mujoco.mjx._src.types.Data, model: mujoco.mjx._src.types.Model, specs: collections.abc.Sequence[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]]):
        ...
    def _slice(self, name: str, idx: typing.Union[int, slice, collections.abc.Sequence[int]]):
        ...
    def set(self, name: str, value: jax.Array) -> mujoco.mjx._src.types.Data:
        """
        Set the value of an array in an MJX Data.
        """
class BindModel:
    """
    Class holding the requested MJX Model and spec id for binding a spec to Model.
    """
    def __getattr__(self, name: str):
        ...
    def __init__(self, model: mujoco.mjx._src.types.Model, specs: collections.abc.Sequence[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]]):
        ...
    def _slice(self, name: str, idx: typing.Union[int, slice, collections.abc.Sequence[int]]):
        ...
def _bind_data(self: mujoco.mjx._src.types.Data, model: mujoco.mjx._src.types.Model, obj: typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin, collections.abc.Iterable[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]]]) -> BindData:
    """
    Bind a Mujoco spec to an MJX Data.
    """
def _bind_model(self: mujoco.mjx._src.types.Model, obj: typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin, collections.abc.Iterable[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]]]) -> BindModel:
    """
    Bind a Mujoco spec to an MJX Model.
    """
def _decode_pyramid(pyramid: jax.Array, mu: jax.Array, condim: int) -> jax.Array:
    """
    Converts pyramid representation to contact force.
    """
def _getadr(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], obj: mujoco._enums.mjtObj) -> numpy.ndarray:
    """
    Gets the name addresses for the given object type.
    """
def _getnum(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], obj: mujoco._enums.mjtObj) -> int:
    """
    Gets the number of objects for the given object type.
    """
def _is_intersect(p1: jax.Array, p2: jax.Array, p3: jax.Array, p4: jax.Array) -> jax.Array:
    """
    Check for intersection between two lines defined by their endpoints.
    """
def _length_circle(p0: jax.Array, p1: jax.Array, ind: jax.Array, rad: jax.Array) -> jax.Array:
    """
    Compute length of circle.
    """
def apply_ft(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, force: jax.Array, torque: jax.Array, point: jax.Array, body_id: jax.Array) -> jax.Array:
    """
    Apply Cartesian force and torque.
    """
def contact_force(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, contact_id: int, to_world_frame: bool = False) -> jax.Array:
    """
    Extract 6D force:torque for one contact, in contact frame by default.
    """
def contact_force_dim(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, dim: int) -> typing.Tuple[jax.Array, numpy.ndarray]:
    """
    Extract 6D force:torque for contacts with dimension dim.
    """
def full_m(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> jax.Array:
    """
    Reconstitute dense mass matrix from qM.
    """
def id2name(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], typ: mujoco._enums.mjtObj, i: int) -> typing.Optional[str]:
    """
    Gets the name of an object with the specified mjtObj type and ids.
    
    See mujoco.id2name for more info.
    
    Args:
      m: mujoco.MjModel or mjx.Model
      typ: mujoco.mjtObj type
      i: the id
    
    Returns:
      the name string, or None if not found
    """
def is_sparse(m: typing.Union[mujoco._structs.MjModel, mujoco.mjx._src.types.Model]) -> bool:
    """
    Return True if this model should create sparse mass matrices.
    
    Args:
      m: a MuJoCo or MJX model
    
    Returns:
      True if provided model should create sparse mass matrices
    
    Modern TPUs have specialized hardware for rapidly operating over sparse
    matrices, whereas GPUs tend to be faster with dense matrices as long as they
    fit onto the device.  As such, the default behavior in MJX (via
    ``JacobianType.AUTO``) is sparse if ``nv`` is >= 60 or MJX detects a TPU as
    the default backend, otherwise dense.
    """
def jac(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, point: jax.Array, body_id: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Compute pair of (NV, 3) Jacobians of global point attached to body.
    """
def jac_dot(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, point: jax.Array, body_id: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Compute pair of (NV, 3) Jacobian time derivatives of global point attached to body.
    """
def local_to_global(world_pos: jax.Array, world_quat: jax.Array, local_pos: jax.Array, local_quat: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Converts local position/orientation to world frame.
    """
def make_m(m: mujoco.mjx._src.types.Model, a: jax.Array, b: jax.Array, d: typing.Optional[jax.Array] = None) -> jax.Array:
    """
    Computes M = a @ b.T + diag(d).
    """
def mul_m(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, vec: jax.Array) -> jax.Array:
    """
    Multiply vector by inertia matrix.
    """
def muscle_bias(length: jax.Array, lengthrange: jax.Array, acc0: jax.Array, prm: jax.Array) -> jax.Array:
    """
    Muscle passive force.
    """
def muscle_dynamics(ctrl: jax.Array, act: jax.Array, prm: jax.Array) -> jax.Array:
    """
    Muscle activation dynamics.
    """
def muscle_dynamics_timescale(dctrl: jax.Array, tau_act: jax.Array, tau_deact: jax.Array, smoothing_width: jax.Array) -> jax.Array:
    """
    Muscle time constant with optional smoothing.
    """
def muscle_gain(length: jax.Array, vel: jax.Array, lengthrange: jax.Array, acc0: jax.Array, prm: jax.Array) -> jax.Array:
    """
    Muscle active force.
    """
def muscle_gain_length(length: jax.Array, lmin: jax.Array, lmax: jax.Array) -> jax.Array:
    """
    Normalized muscle length-gain curve.
    """
def name2id(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], typ: mujoco._enums.mjtObj, name: str) -> int:
    """
    Gets the id of an object with the specified mjtObj type and name.
    
    See mujoco.mj_name2id for more info.
    
    Args:
      m: mujoco.MjModel or mjx.Model
      typ: mujoco.mjtObj type
      name: the name of the object
    
    Returns:
     the id, or -1 if not found
    """
def wrap(x0: jax.Array, x1: jax.Array, xpos: jax.Array, xmat: jax.Array, size: jax.Array, side: jax.Array, sidesite: jax.Array, is_sphere: jax.Array, is_wrap_inside: bool, wrap_inside_maxiter: int, wrap_inside_tolerance: float, wrap_inside_z_init: float) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Wrap tendon around sphere or cylinder.
    """
def wrap_circle(d: jax.Array, sd: jax.Array, sidesite: jax.Array, rad: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Compute circle wrap arc length and end points.
    """
def wrap_inside(end: jax.Array, radius: jax.Array, maxiter: int, tolerance: float, z_init: float) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Compute 2D inside wrap point.
    
    Args:
      end: 2D points
      radius: radius of circle
    
    Returns:
      status: 0 if wrap, else -1
      concatentated 2D wrap points: jax.Array
    """
def xfrc_accumulate(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> jax.Array:
    """
    Accumulate xfrc_applied into a qfrc.
    """
