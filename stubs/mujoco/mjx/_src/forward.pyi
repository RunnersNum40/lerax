"""
Forward step functions.
"""
from __future__ import annotations

import functools as functools

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from jax import numpy as jp
from mujoco.mjx import warp as mjxw
from mujoco.mjx._src import (
    collision_driver,
    constraint,
    derivative,
    math,
    passive,
    scan,
    sensor,
    smooth,
    solver,
    support,
)
from mujoco.mjx._src.types import (
    BiasType,
    Data,
    DataJAX,
    DisableBit,
    DynType,
    GainType,
    Impl,
    IntegratorType,
    JointType,
    Model,
    ModelJAX,
    TrnType,
)

__all__: list[str] = ['BiasType', 'Data', 'DataJAX', 'DisableBit', 'DynType', 'GainType', 'Impl', 'IntegratorType', 'JointType', 'Model', 'ModelJAX', 'TrnType', 'collision_driver', 'constraint', 'derivative', 'euler', 'forward', 'functools', 'fwd_acceleration', 'fwd_actuation', 'fwd_position', 'fwd_velocity', 'implicit', 'jax', 'jp', 'math', 'mjxw', 'mujoco', 'named_scope', 'np', 'passive', 'rungekutta4', 'scan', 'sensor', 'smooth', 'solver', 'step', 'support']
def _advance(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Advance state and time given activation derivatives and acceleration.
    """
def _integrate_pos(*args, **kwargs) -> jax.Array:
    """
    Integrate position given velocity.
    """
def _next_activation(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, act_dot: jax.Array) -> jax.Array:
    """
    Returns the next act given the current act_dot, after clamping.
    """
def euler(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Euler integrator, semi-implicit in velocity.
    """
def forward(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Forward dynamics.
    """
def fwd_acceleration(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Add up all non-constraint forces, compute qacc_smooth.
    """
def fwd_actuation(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Actuation-dependent computations.
    """
def fwd_position(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Position-dependent computations.
    """
def fwd_velocity(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Velocity-dependent computations.
    """
def implicit(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Integrates fully implicit in velocity.
    """
def named_scope(fn, name: str = ''):
    ...
def rungekutta4(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Runge-Kutta explicit order 4 integrator.
    """
def step(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Advance simulation.
    """
_RK4_A: numpy.ndarray  # value = array([[0.5, 0. , 0. ],...
_RK4_B: numpy.ndarray  # value = array([0.16666667, 0.33333333, 0.33333333, 0.16666667])
