from __future__ import annotations

from typing import Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Box

from .base_wrapper import AbstractWrapper
from .utils import rescale_box


class AbstractPureTransformActionWrapper[
    WrapperActType, StateType: AbstractEnvLikeState, ActType, ObsType
](
    AbstractWrapper[
        StateType,
        WrapperActType,
        ObsType,
        StateType,
        ActType,
        ObsType,
    ]
):
    """
    Base class for wrappers that apply a pure function to the action before passing it to
    the environment
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType]]
    func: eqx.AbstractVar[Callable[[WrapperActType], ActType]]
    action_space: eqx.AbstractVar[AbstractSpace[WrapperActType]]

    def reset(self, *, key: Key) -> tuple[StateType, ObsType, dict]:
        return self.env.reset(key=key)

    def step(self, state: StateType, action: WrapperActType, *, key: Key):
        return self.env.step(state, self.func(action), key=key)

    def render(self, state: StateType):
        self.env.render(state)

    def close(self):
        self.env.close()


class TransformAction[
    WrapperActType, StateType: AbstractEnvLikeState, ActType, ObsType
](AbstractPureTransformActionWrapper[WrapperActType, StateType, ActType, ObsType]):
    """Apply a function to the action before passing it to the environment"""

    env: AbstractEnvLike[StateType, ActType, ObsType]
    func: Callable[[WrapperActType], ActType]
    action_space: AbstractSpace[WrapperActType]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType],
        func: Callable[[WrapperActType], ActType],
        action_space: AbstractSpace[WrapperActType],
    ):
        self.env = env
        self.func = func
        self.action_space = action_space


class ClipAction[StateType: AbstractEnvLikeState, ObsType](
    AbstractPureTransformActionWrapper[
        Float[Array, " ..."], StateType, Float[Array, " ..."], ObsType
    ],
):
    """
    Clips every action to the environment's action space.
    """

    env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    action_space: Box

    def __init__(self, env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType]):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                "ClipAction only supports `Box` action spaces "
                f"not {type(env.action_space)}"
            )

        def clip(action: Float[Array, " ..."]) -> Float[Array, " ..."]:
            assert isinstance(env.action_space, Box)
            return jnp.clip(action, env.action_space.low, env.action_space.high)

        action_space = Box(-jnp.inf, jnp.inf, shape=env.action_space.shape)

        self.env = env
        self.func = clip
        self.action_space = action_space


class RescaleAction[StateType: AbstractEnvLikeState, ObsType](
    AbstractPureTransformActionWrapper[
        Float[Array, " ..."], StateType, Float[Array, " ..."], ObsType
    ],
):
    """Affinely rescale a box action to a different range"""

    env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    action_space: Box

    def __init__(
        self,
        env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType],
        min: Float[Array, " ..."] = jnp.array(-1.0),
        max: Float[Array, " ..."] = jnp.array(1.0),
    ):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                "RescaleActiononly supports `Box` action spaces"
                f" not {type(env.action_space)}"
            )

        action_space, _, rescale = rescale_box(env.action_space, min, max)

        self.env = env
        self.func = rescale
        self.action_space = action_space
