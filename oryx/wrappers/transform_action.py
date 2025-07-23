from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from oryx.env import AbstractEnvLike
from oryx.spaces import AbstractSpace, Box

from .base_wrapper import AbstractActionWrapper
from .utils import rescale_box


class AbstractTransformActionWrapper[WrapperActType, ActType, ObsType](
    AbstractActionWrapper[WrapperActType, ActType, ObsType]
):
    """
    Base class for wrappers that apply a function to the action before passing it to
    the environment
    """

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[WrapperActType], ActType]]
    _action_space: eqx.AbstractVar[AbstractSpace[WrapperActType]]

    def action(
        self, state: eqx.nn.State, action: WrapperActType, *, key: Key | None
    ) -> tuple[eqx.nn.State, ActType]:
        transformed_action = self.func(action)
        return state, transformed_action

    @property
    def action_space(self) -> AbstractSpace[WrapperActType]:
        return self._action_space


class TransformActionWrapper[WrapperActType, ActType, ObsType](
    AbstractActionWrapper[WrapperActType, ActType, ObsType]
):
    """Apply a function to the action before passing it to the environment"""

    env: AbstractEnvLike[ActType, ObsType]
    func: Callable[[WrapperActType], ActType]
    _action_space: AbstractSpace[WrapperActType]

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        func: Callable[[WrapperActType], ActType],
        action_space: AbstractSpace[WrapperActType],
    ):
        self.env = env
        self.func = func
        self._action_space = action_space


class ClipActionWrapper[ObsType](
    AbstractTransformActionWrapper[Float[Array, " ..."], Float[Array, " ..."], ObsType],
):
    """
    Clip the action to be within the environment's action_space before passing it to
    the environment
    """

    env: AbstractEnvLike[Float[Array, " ..."], ObsType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    _action_space: Box

    def __init__(self, env: AbstractEnvLike[Float[Array, " ..."], ObsType]):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                f"Clip action wrapper only works with Box action spaces not {type(env.action_space)}"
            )

        def clip(action: Float[Array, " ..."]) -> Float[Array, " ..."]:
            assert isinstance(env.action_space, Box)
            return jnp.clip(action, env.action_space.low, env.action_space.high)

        action_space = Box(-jnp.inf, jnp.inf, shape=env.action_space.shape)

        self.env = env
        self.func = clip
        self._action_space = action_space


class RescaleActionWrapper[ObsType](
    AbstractTransformActionWrapper[Float[Array, " ..."], Float[Array, " ..."], ObsType],
):
    """Affinely rescale a box action to a different range"""

    env: AbstractEnvLike[Float[Array, " ..."], ObsType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    _action_space: Box

    def __init__(
        self,
        env: AbstractEnvLike[Float[Array, " ..."], ObsType],
        min: Float[Array, " ..."] = jnp.array(-1.0),
        max: Float[Array, " ..."] = jnp.array(1.0),
    ):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                f"Clip action wrapper only works with Box action spaces not {type(env.action_space)}"
            )

        action_space, rescale, _ = rescale_box(env.action_space, min, max)

        self.env = env
        self.func = rescale
        self._action_space = action_space
