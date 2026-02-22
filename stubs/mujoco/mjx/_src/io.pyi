"""
Functions to initialize, load, or save data.
"""

from __future__ import annotations

import copy as copy
import logging as logging
import os as os
import typing
import warnings as warnings
from typing import Any

import jax as jax
import jaxlib._jax
import mujoco as mujoco
import numpy
import numpy as np
import scipy as scipy
from jax import numpy as jp
from jax.extend import backend
from mujoco.mjx import warp as mjxw
from mujoco.mjx._src import collision_driver, constraint, mesh, support, types

__all__: list[str] = [
    "Any",
    "backend",
    "collision_driver",
    "constraint",
    "copy",
    "get_data",
    "get_data_into",
    "has_cuda_gpu_device",
    "jax",
    "jp",
    "logging",
    "make_data",
    "mesh",
    "mjwp",
    "mjwp_types",
    "mjxw",
    "mujoco",
    "np",
    "os",
    "put_data",
    "put_model",
    "scipy",
    "support",
    "types",
    "warnings",
    "wp",
]

def _check_impl_device_compatibility(
    impl: typing.Union[str, mujoco.mjx._src.types.Impl], device: jaxlib._jax.Device
) -> None:
    """
    Checks that the implementation is compatible with the device.
    """

def _get_contact(c: mujoco._structs._MjContactList, cx: mujoco.mjx._src.types.Contact):
    """
    Converts mjx.Contact to mujoco._structs._MjContactList.
    """

def _get_data_into(
    result: typing.Union[mujoco._structs.MjData, typing.List[mujoco._structs.MjData]],
    m: mujoco._structs.MjModel,
    d: mujoco.mjx._src.types.Data,
):
    """
    Gets mjx.Data from a device into an existing mujoco.MjData or list.
    """

def _get_data_into_warp(
    result: typing.Union[mujoco._structs.MjData, typing.List[mujoco._structs.MjData]],
    m: mujoco._structs.MjModel,
    d: mujoco.mjx._src.types.Data,
):
    """
    Gets mjx.Data from a device into an existing mujoco.MjData or list.
    """

def _get_nested_attr(obj: typing.Any, attr_name: str, split: str) -> typing.Any:
    """
    Returns the nested attribute from an object.
    """

def _is_cuda_gpu_device(device: jaxlib._jax.Device) -> bool: ...
def _make_data_c(
    m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel],
    device: typing.Optional[jaxlib._jax.Device] = None,
) -> mujoco.mjx._src.types.Data:
    """
    Allocate and initialize Data for the C implementation.
    """

def _make_data_contact_jax(
    condim: numpy.ndarray, efc_address: numpy.ndarray
) -> mujoco.mjx._src.types.Contact:
    """
    Create contact for the Data object.
    """

def _make_data_jax(
    m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel],
    device: typing.Optional[jaxlib._jax.Device] = None,
) -> mujoco.mjx._src.types.Data:
    """
    Allocate and initialize Data for the JAX implementation.
    """

def _make_data_public_fields(
    m: mujoco.mjx._src.types.Model,
) -> typing.Dict[str, typing.Any]:
    """
    Create public fields for the Data object.
    """

def _make_data_warp(
    m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel],
    device: typing.Optional[jaxlib._jax.Device] = None,
    nconmax: typing.Optional[int] = None,
    njmax: typing.Optional[int] = None,
) -> mujoco.mjx._src.types.Data:
    """
    Allocate and initialize Data for the Warp implementation.
    """

def _put_contact(
    c: mujoco._structs._MjContactList, dim: numpy.ndarray, efc_address: numpy.ndarray
) -> typing.Tuple[mujoco.mjx._src.types.Contact, numpy.ndarray]:
    """
    Converts mujoco.structs._MjContactList into mjx.Contact.
    """

def _put_data_c(
    m: mujoco._structs.MjModel,
    d: mujoco._structs.MjData,
    device: typing.Optional[jaxlib._jax.Device] = None,
) -> mujoco.mjx._src.types.Data:
    """
    Puts mujoco.MjData onto a device, resulting in mjx.Data.
    """

def _put_data_jax(
    m: mujoco._structs.MjModel,
    d: mujoco._structs.MjData,
    device: typing.Optional[jaxlib._jax.Device] = None,
) -> mujoco.mjx._src.types.Data:
    """
    Puts mujoco.MjData onto a device, resulting in mjx.Data.
    """

def _put_data_public_fields(d: mujoco._structs.MjData) -> typing.Dict[str, typing.Any]:
    """
    Returns public fields from mujoco.MjData in a dictionary.
    """

def _put_model_c(
    m: mujoco._structs.MjModel, device: typing.Optional[jaxlib._jax.Device] = None
) -> mujoco.mjx._src.types.Model:
    """
    Puts mujoco.MjModel onto a device, resulting in mjx.Model.
    """

def _put_model_jax(
    m: mujoco._structs.MjModel, device: typing.Optional[jaxlib._jax.Device] = None
) -> mujoco.mjx._src.types.Model:
    """
    Puts mujoco.MjModel onto a device, resulting in mjx.Model.
    """

def _put_model_warp(
    m: mujoco._structs.MjModel, device: typing.Optional[jaxlib._jax.Device] = None
) -> mujoco.mjx._src.types.Model:
    """
    Puts mujoco.MjModel onto a device, resulting in mjx.Model.
    """

def _put_option(
    o: mujoco._structs.MjOption,
    impl: mujoco.mjx._src.types.Impl,
    impl_fields: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> mujoco.mjx._src.types.Option:
    """
    Returns mjx.Option given mujoco.MjOption.
    """

def _put_statistic(
    s: mujoco._structs.MjStatistic, impl: mujoco.mjx._src.types.Impl
) -> typing.Union[mujoco.mjx._src.types.Statistic, mujoco.mjx._src.types.StatisticWarp]:
    """
    Puts mujoco.MjStatistic onto a device, resulting in mjx.Statistic.
    """

def _resolve_device(impl: mujoco.mjx._src.types.Impl) -> jaxlib._jax.Device:
    """
    Resolves a device based on the implementation.
    """

def _resolve_impl(device: jaxlib._jax.Device) -> mujoco.mjx._src.types.Impl:
    """
    Pick a default implementation based on the device specified.
    """

def _resolve_impl_and_device(
    impl: typing.Union[str, mujoco.mjx._src.types.Impl, NoneType],
    device: typing.Optional[jaxlib._jax.Device] = None,
) -> typing.Tuple[mujoco.mjx._src.types.Impl, jaxlib._jax.Device]:
    """
    Resolves a implementation and device.
    """

def _strip_weak_type(tree): ...
def _wp_to_np_type(wp_field: typing.Any, name: str = "") -> typing.Any:
    """
    Converts a warp type to an MJX compatible numpy type.
    """

def get_data(
    m: mujoco._structs.MjModel, d: mujoco.mjx._src.types.Data
) -> typing.Union[mujoco._structs.MjData, typing.List[mujoco._structs.MjData]]:
    """
    Gets mjx.Data from a device, resulting in mujoco.MjData or List[MjData].
    """

def get_data_into(
    result: typing.Union[mujoco._structs.MjData, typing.List[mujoco._structs.MjData]],
    m: mujoco._structs.MjModel,
    d: mujoco.mjx._src.types.Data,
):
    """
    Gets mjx.Data from a device into an existing mujoco.MjData or list.
    """

def has_cuda_gpu_device() -> bool: ...
def make_data(
    m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel],
    device: typing.Optional[jaxlib._jax.Device] = None,
    impl: typing.Union[str, mujoco.mjx._src.types.Impl, NoneType] = None,
    _full_compat: bool = False,
    nconmax: typing.Optional[int] = None,
    njmax: typing.Optional[int] = None,
) -> mujoco.mjx._src.types.Data:
    """
    Allocate and initialize Data.

    Args:
      m: the model to use
      device: which device to use - if unspecified picks the default device
      impl: implementation to use ('jax', 'warp')
      nconmax: maximum number of contacts to allocate for warp across all worlds
        Since the number of worlds is **not** pre-defined in JAX, we use the
        `nconmax` argument to set the upper bound for the number of contacts
        across all worlds. In MuJoCo Warp, the analgous field is called
        `naconmax`.
      njmax: maximum number of constraints to allocate for warp across all worlds

    Returns:
      an initialized mjx.Data placed on device

    Raises:
      ValueError: if the model's impl does not match the make_data impl
      NotImplementedError: if the impl is not implemented yet
    """

def put_data(
    m: mujoco._structs.MjModel,
    d: mujoco._structs.MjData,
    device: typing.Optional[jaxlib._jax.Device] = None,
    impl: typing.Union[str, mujoco.mjx._src.types.Impl, NoneType] = None,
    nconmax: int = -1,
    njmax: int = -1,
) -> mujoco.mjx._src.types.Data:
    """
    Puts mujoco.MjData onto a device, resulting in mjx.Data.

    Args:
      m: the model to use
      d: the data to put on device
      device: which device to use - if unspecified picks the default device
      impl: implementation to use ('jax', 'warp')
      nconmax: maximum number of contacts to allocate for warp
      njmax: maximum number of constraints to allocate for warp

    Returns:
      an mjx.Data placed on device
    """

def put_model(
    m: mujoco._structs.MjModel,
    device: typing.Optional[jaxlib._jax.Device] = None,
    impl: typing.Union[str, mujoco.mjx._src.types.Impl, NoneType] = None,
) -> mujoco.mjx._src.types.Model:
    """
    Puts mujoco.MjModel onto a device, resulting in mjx.Model.

    Args:
      m: the model to put onto device
      device: which device to use - if unspecified picks the default device
      impl: implementation to use

    Returns:
      an mjx.Model placed on device

    Raises:
      ValueError: if impl is not supported
    """

mjwp = None
mjwp_types = None
wp = None
