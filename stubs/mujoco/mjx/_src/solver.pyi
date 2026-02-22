"""
Constraint solvers.
"""

from __future__ import annotations

import dataclasses
import typing

import jax as jax
import mujoco as mujoco
from jax import numpy as jp
from mujoco.mjx._src import math, smooth, support
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src.types import (
    ConeType,
    Data,
    DataJAX,
    DisableBit,
    Model,
    ModelJAX,
    OptionJAX,
    SolverType,
)

__all__: list[str] = [
    "ConeType",
    "Context",
    "Data",
    "DataJAX",
    "DisableBit",
    "Model",
    "ModelJAX",
    "OptionJAX",
    "PyTreeNode",
    "SolverType",
    "jax",
    "jp",
    "math",
    "mujoco",
    "smooth",
    "solve",
    "support",
]

class Context(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Data updated during each solver iteration.

    Attributes:
      qacc: acceleration (from Data)                    (nv,)
      qfrc_constraint: constraint force (from Data)     (nv,)
      Jaref: Jac*qacc - aref                            (nefc,)
      efc_force: constraint force in constraint space   (nefc,)
      Ma: M*qacc                                        (nv,)
      grad: gradient of master cost                     (nv,)
      Mgrad: M / grad                                   (nv,)
      search: linesearch vector                         (nv,)
      gauss: gauss Cost
      cost: constraint + Gauss cost
      prev_cost: cost from previous iter
      solver_niter: number of solver iterations
      active: active (quadratic) constraints            (nefc,)
      fri: friction of regularized cone                 (num(con.dim > 1), 6)
      dm: regularized constraint mass                   (num(con.dim > 1))
      u: friction cone (normal and tangents)            (num(con.dim > 1), 6)
      h: cone hessian                                   (num(con.dim > 1), 6, 6)
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'qacc': Field(name='qacc',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'qfrc_constraint': Field(name='qfrc_constraint',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'Jaref': Field(name='Jaref',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'efc_force': Field(name='efc_force',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'Ma': Field(name='Ma',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'grad': Field(name='grad',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'Mgrad': Field(name='Mgrad',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'search': Field(name='search',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'gauss': Field(name='gauss',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cost': Field(name='cost',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'prev_cost': Field(name='prev_cost',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'solver_niter': Field(name='solver_niter',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'active': Field(name='active',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'fri': Field(name='fri',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'dm': Field(name='dm',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'u': Field(name='u',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'h': Field(name='h',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = (
        "qacc",
        "qfrc_constraint",
        "Jaref",
        "efc_force",
        "Ma",
        "grad",
        "Mgrad",
        "search",
        "gauss",
        "cost",
        "prev_cost",
        "solver_niter",
        "active",
        "fri",
        "dm",
        "u",
        "h",
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
    @classmethod
    def create(
        cls,
        m: mujoco.mjx._src.types.Model,
        d: mujoco.mjx._src.types.Data,
        grad: bool = True,
    ) -> Context: ...
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self,
        qacc: jax.Array,
        qfrc_constraint: jax.Array,
        Jaref: jax.Array,
        efc_force: jax.Array,
        Ma: jax.Array,
        grad: jax.Array,
        Mgrad: jax.Array,
        search: jax.Array,
        gauss: jax.Array,
        cost: jax.Array,
        prev_cost: jax.Array,
        solver_niter: jax.Array,
        active: jax.Array,
        fri: jax.Array,
        dm: jax.Array,
        u: jax.Array,
        h: jax.Array,
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class _LSContext(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Data updated during each line search iteration.

    Attributes:
      lo: low point bounding the line search interval
      hi: high point bounding the line search interval
      swap: True if low or hi was swapped in the line search iteration
      ls_iter: number of linesearch iterations
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'lo': Field(name='lo',type=<class 'mujoco.mjx._src.solver._LSPoint'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'hi': Field(name='hi',type=<class 'mujoco.mjx._src.solver._LSPoint'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'swap': Field(name='swap',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'ls_iter': Field(name='ls_iter',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ("lo", "hi", "swap", "ls_iter")
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
        self, lo: _LSPoint, hi: _LSPoint, swap: jax.Array, ls_iter: jax.Array
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

class _LSPoint(mujoco.mjx._src.dataclasses.PyTreeNode):
    """
    Line search evaluation point.

    Attributes:
      alpha: step size that reduces f(x + alpha * p) given search direction p
      cost: line search cost
      deriv_0: first derivative of quadratic
      deriv_1: second derivative of quadratic
    """

    __dataclass_fields__: typing.ClassVar[
        dict
    ]  # value = {'alpha': Field(name='alpha',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'cost': Field(name='cost',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'deriv_0': Field(name='deriv_0',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD), 'deriv_1': Field(name='deriv_1',type=<class 'jax.Array'>,default=<dataclasses._MISSING_TYPE object>,default_factory=<dataclasses._MISSING_TYPE object>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=False,_field_type=_FIELD)}
    __dataclass_params__: typing.ClassVar[
        dataclasses._DataclassParams
    ]  # value = _DataclassParams(init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=True,match_args=True,kw_only=False,slots=False,weakref_slot=False)
    __match_args__: typing.ClassVar[tuple] = ("alpha", "cost", "deriv_0", "deriv_1")
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
    @classmethod
    def create(
        cls,
        m: mujoco.mjx._src.types.Model,
        d: mujoco.mjx._src.types.Data,
        ctx: Context,
        alpha: jax.Array,
        jv: jax.Array,
        quad: jax.Array,
        quad_gauss: jax.Array,
        uu: jax.Array,
        v0: jax.Array,
        uv: jax.Array,
        vv: jax.Array,
    ) -> _LSPoint:
        """
        Creates a linesearch point with first and second derivatives.
        """
    def __delattr__(self, name): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __init__(
        self, alpha: jax.Array, cost: jax.Array, deriv_0: jax.Array, deriv_1: jax.Array
    ) -> None: ...
    def __replace__(self, **changes): ...
    def __repr__(self): ...
    def __setattr__(self, name, value): ...

def _linesearch(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, ctx: Context
) -> Context:
    """
    Performs a zoom linesearch to find optimal search step size.

    Args:
      m: model defining search options and other needed terms
      d: data with inertia matrix and other needed terms
      ctx: current solver context

    Returns:
      updated context with new qacc, Ma, Jaref
    """

def _rescale(m: mujoco.mjx._src.types.Model, value: jax.Array) -> jax.Array: ...
def _update_constraint(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, ctx: Context
) -> Context:
    """
    Updates constraint force and resulting cost given last solver iteration.

    Corresponds to CGupdateConstraint in mujoco/src/engine/engine_solver.c

    Args:
      m: model defining constraints
      d: data which contains latest qacc and smooth terms
      ctx: current solver context

    Returns:
      context with new constraint force and costs
    """

def _update_gradient(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, ctx: Context
) -> Context:
    """
    Updates grad and M / grad given latest solver iteration.

    Corresponds to CGupdateGradient in mujoco/src/engine/engine_solver.c

    Args:
      m: model defining constraints
      d: data which contains latest smooth terms
      ctx: current solver context

    Returns:
      context with new grad and M / grad
    Raises:
      NotImplementedError: for unsupported solver type
    """

def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
    """
    Scan-based implementation (jit ok, reverse-mode autodiff ok).
    """

def solve(
    m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data
) -> mujoco.mjx._src.types.Data:
    """
    Finds forces that satisfy constraints using conjugate gradient descent.
    """
