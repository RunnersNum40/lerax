"""
Core smooth dynamics functions.
"""
from __future__ import annotations

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from jax import numpy as jp
from mujoco.mjx import warp as mjxw
from mujoco.mjx._src import math, scan, support
from mujoco.mjx._src.types import (
    CamLightType,
    Data,
    DataJAX,
    DisableBit,
    EqType,
    Impl,
    JointType,
    Model,
    ModelJAX,
    ObjType,
    TrnType,
    WrapType,
)

__all__: list[str] = ['CamLightType', 'Data', 'DataJAX', 'DisableBit', 'EqType', 'Impl', 'JointType', 'Model', 'ModelJAX', 'ObjType', 'TrnType', 'WrapType', 'camlight', 'com_pos', 'com_vel', 'crb', 'factor_m', 'jax', 'jp', 'kinematics', 'math', 'mjxw', 'mujoco', 'np', 'rne', 'rne_postconstraint', 'scan', 'solve_m', 'subtree_vel', 'support', 'tendon', 'tendon_armature', 'tendon_bias', 'tendon_dot', 'transmission']
def _site_dof_mask(m: mujoco.mjx._src.types.Model) -> numpy.ndarray:
    """
    Creates a dof mask for site transmissions.
    """
def camlight(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes camera and light positions and orientations.
    """
def com_pos(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Maps inertias and motion dofs to global frame centered at subtree-CoM.
    """
def com_vel(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes cvel, cdof_dot.
    """
def crb(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Runs composite rigid body inertia algorithm.
    """
def factor_m(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Gets factorizaton of inertia-like matrix M, assumed spd.
    """
def kinematics(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Converts position/velocity from generalized coordinates to maximal.
    """
def rne(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, flg_acc: bool = False) -> mujoco.mjx._src.types.Data:
    """
    Computes inverse dynamics using the recursive Newton-Euler algorithm.
    
    flg_acc=False removes inertial term.
    """
def rne_postconstraint(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    RNE with complete data: compute cacc, cfrc_ext, cfrc_int.
    """
def solve_m(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, x: jax.Array) -> jax.Array:
    """
    Computes sparse backsubstitution:  x = inv(L'*D*L)*y .
    """
def subtree_vel(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Subtree linear velocity and angular momentum.
    """
def tendon(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes tendon lengths and moments.
    """
def tendon_armature(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Add tendon armature to qM.
    """
def tendon_bias(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Add bias force due to tendon armature.
    """
def tendon_dot(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> jax.Array:
    """
    Compute time derivative of dense tendon Jacobian for one tendon.
    """
def transmission(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes actuator/transmission lengths and moments.
    """
