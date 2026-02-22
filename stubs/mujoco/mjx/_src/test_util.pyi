"""
Utilities for testing.
"""

from __future__ import annotations

import os as os
import sys as sys
import time as time
from xml.etree import ElementTree as ET

import jax as jax
import mujoco as mujoco
import numpy
import numpy as np
from etils import epath
from mujoco.mjx._src import forward, io
from mujoco.mjx._src.types import Data

__all__: list[str] = [
    "Data",
    "ET",
    "benchmark",
    "create_mjcf",
    "efc_order",
    "epath",
    "forward",
    "io",
    "jax",
    "load_test_file",
    "mujoco",
    "np",
    "os",
    "p",
    "sys",
    "time",
]

def _geom_solparams(
    pair: bool = False, enable_contact: bool = True
) -> typing.Dict[str, str]:
    """
    Returns geom solver parameters.
    """

def _make_actuator(
    actuator_type: str,
    joint: typing.Optional[str] = None,
    site: typing.Optional[str] = None,
    refsite: typing.Optional[str] = None,
) -> typing.Dict[str, str]:
    """
    Returns attributes for an actuator.
    """

def _make_geom(
    pos: str, size: float, name: str, enable_contact: bool = True
) -> typing.Dict[str, str]:
    """
    Returns attributes for a sphere geom.
    """

def _make_joint(joint_type: str, name: str) -> typing.Dict[str, str]:
    """
    Returns attributes for a joint.
    """

def _measure(fn, *args) -> typing.Tuple[float, float]:
    """
    Reports jit time and op time for a function.
    """

def benchmark(
    m: mujoco._structs.MjModel,
    nstep: int = 1000,
    batch_size: int = 1024,
    unroll_steps: int = 1,
    solver: str = "newton",
    iterations: int = 1,
    ls_iterations: int = 4,
) -> typing.Tuple[float, float, int]:
    """
    Benchmark a model.
    """

def create_mjcf(
    seed: int,
    min_trees: int = 1,
    max_trees: int = 1,
    max_tree_depth: int = 5,
    body_pos: typing.Tuple[float, float, float] = (0.0, 0.0, -0.5),
    geom_pos: typing.Tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_stacked_joints=4,
    max_geoms_per_body=2,
    max_contact_excludes=1,
    max_contact_pairs=4,
    disable_actuation_pct: int = 0,
    add_actuators: bool = False,
    root_always_free: bool = False,
    enable_contact: bool = True,
) -> str:
    """
    Creates a random MJCF for testing.

    Args:
      seed: seed for rng
      min_trees: minimum number of kinematic trees to generate
      max_trees: maximum number of kinematic trees to generate
      max_tree_depth: the maximum tree depth
      body_pos: the default body position relative to the parent
      geom_pos: the default geom position in the body frame
      max_stacked_joints: maximum number of joints to stack for each body
      max_geoms_per_body: maximum number of geoms per body
      max_contact_excludes: maximum number of bodies to exlude from contact
      max_contact_pairs: maximum number of explicit geom contact pairs in the xml
      disable_actuation_pct: the percentage of time to disable actuation via the
        disable flag
      add_actuators: whether to add actuators
      root_always_free: if True, the root body of each kinematic tree has a free
        joint with the world
      enable_contact: if False, disables all contacts via contype/conaffinity

    Returns:
      an XML string for the MuJoCo config
    Raises:
      AssertionError when args are not in the correct ranges
    """

def efc_order(
    m: mujoco._structs.MjModel,
    d: mujoco._structs.MjData,
    dx: mujoco.mjx._src.types.Data,
) -> numpy.ndarray:
    """
    Returns a sort order such that dx.efc_*[order][:d._impl.nefc] == d.efc_*.
    """

def load_test_file(name: str) -> mujoco._structs.MjModel:
    """
    Loads a mujoco.MjModel based on the file name.
    """

def p(pct: int) -> bool: ...

_ACTUATOR_TYPES: list = ["motor", "velocity", "position", "general", "intvelocity"]
_DIMS: list = ["3"]
_DYN_PRMS: list = ["0.189", "2.1"]
_DYN_TYPES: list = ["none", "integrator", "filter", "filterexact"]
_FRICTIONS: list = ["1.2 0.003 0.0002", "0.2 0.0001 0.0005"]
_GAPS: list = ["0.0", "0.005"]
_GEARS: list = ["2.1 0.0 3.3 0 2.3 0", "5.0 3.1 0 2.3 0.0 1.1"]
_JOINT_AXES: list = ["1 0 0", "0 1 0", "0 0 1"]
_JOINT_TYPES: list = ["free", "hinge", "slide", "ball"]
_KP_INTVEL: list = ["10000", "2000"]
_KP_POS: list = ["1", "2"]
_KV_VEL: list = ["12", "1", "0", "0.1"]
_MARGINS: list = ["0.0", "0.01", "0.02"]
_PAIR_FRICTIONS: list = ["1.2 0.9 0.003 0.0002 0.0001"]
_SOLIMPS: list = [
    "0.75 0.94 0.002 0.2 2",
    "0.8 0.99 0.001 0.3 6",
    "0.6 0.9 0.003 0.1 1",
]
_SOLREFS: list = ["0.04 1.01", "0.05 1.02", "0.03 1.1", "0.015 1.0"]
