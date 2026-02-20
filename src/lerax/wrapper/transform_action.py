from typing import Any, Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Box

from .base_wrapper import AbstractWrapper, AbstractWrapperState
from .utils import rescale_box


class TransformActionState[StateType: AbstractEnvLikeState](
    AbstractWrapperState[StateType]
):
    env_state: StateType


class AbstractPureTransformActionWrapper[
    WrapperActType,
    WrapperMaskType,
    StateType: AbstractEnvLikeState,
    ActType,
    ObsType,
    MaskType,
](
    AbstractWrapper[
        TransformActionState[StateType],
        WrapperActType,
        ObsType,
        WrapperMaskType,
        StateType,
        ActType,
        ObsType,
        MaskType,
    ]
):
    """
    Base class for wrappers that apply a pure function to the action before passing it to
    the environment.

    Attributes:
        env: The environment to wrap.
        func: The function to apply to the action.
        action_space: The action space of the wrapper.
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType, MaskType]]
    func: eqx.AbstractVar[Callable[[WrapperActType], ActType]]
    mask_func: eqx.AbstractVar[Callable[[MaskType], WrapperMaskType]]
    action_space: eqx.AbstractVar[AbstractSpace[WrapperActType, WrapperMaskType]]

    @property
    def observation_space(self) -> AbstractSpace[ObsType, Any]:
        return self.env.observation_space

    def initial(self, *, key: Key[Array, ""]) -> TransformActionState[StateType]:
        return TransformActionState(self.env.initial(key=key))

    def action_mask(
        self, state: TransformActionState[StateType], *, key: Key[Array, ""]
    ) -> WrapperMaskType | None:
        env_mask = self.env.action_mask(state.env_state, key=key)

        if env_mask is None:
            return None
        else:
            return self.mask_func(env_mask)

    def transition(
        self,
        state: TransformActionState[StateType],
        action: WrapperActType,
        *,
        key: Key[Array, ""],
    ) -> TransformActionState[StateType]:
        return TransformActionState(
            self.env.transition(state.env_state, self.func(action), key=key)
        )

    def observation(
        self, state: TransformActionState[StateType], *, key: Key[Array, ""]
    ) -> ObsType:
        return self.env.observation(state.env_state, key=key)

    def reward(
        self,
        state: TransformActionState[StateType],
        action: WrapperActType,
        next_state: TransformActionState[StateType],
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        return self.env.reward(
            state.env_state, self.func(action), next_state.env_state, key=key
        )

    def terminal(
        self, state: TransformActionState[StateType], *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return self.env.terminal(state.env_state, key=key)

    def truncate(self, state: TransformActionState[StateType]) -> Bool[Array, ""]:
        return self.env.truncate(state.env_state)

    def state_info(self, state: TransformActionState[StateType]) -> dict:
        return self.env.state_info(state.env_state)

    def transition_info(
        self,
        state: TransformActionState[StateType],
        action: WrapperActType,
        next_state: TransformActionState[StateType],
    ) -> dict:
        return self.env.transition_info(
            state.env_state, self.func(action), next_state.env_state
        )


class TransformAction[
    WrapperActType,
    WrapperMaskType,
    StateType: AbstractEnvLikeState,
    ActType,
    ObsType,
    MaskType,
](
    AbstractPureTransformActionWrapper[
        WrapperActType, WrapperMaskType, StateType, ActType, ObsType, MaskType
    ]
):
    """
    Apply a function to the action before passing it to the environment.

    Attributes:
        env: The environment to wrap.
        func: The function to apply to the action.
        action_space: The action space of the wrapper.

    Args:
        env: The environment to wrap.
        func: The function to apply to the action.
        action_space: The action space of the wrapper.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType, MaskType]
    func: Callable[[WrapperActType], ActType]
    mask_func: Callable[[MaskType], WrapperMaskType]
    action_space: AbstractSpace[WrapperActType, WrapperMaskType]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType, MaskType],
        func: Callable[[WrapperActType], ActType],
        action_space: AbstractSpace[WrapperActType, WrapperMaskType],
        mask_func: Callable[[MaskType], WrapperMaskType] = lambda x: x,
    ):
        self.env = env
        self.func = func
        self.mask_func = mask_func
        self.action_space = action_space


class ClipAction[StateType: AbstractEnvLikeState, ObsType, MaskType](
    AbstractPureTransformActionWrapper[
        Float[Array, " ..."],
        MaskType,
        StateType,
        Float[Array, " ..."],
        ObsType,
        MaskType,
    ]
):
    """
    Clips every action to the environment's action space.

    Note:
        Only compatible with `Box` action spaces.

    Attributes:
        env: The environment to wrap.
        action_space: The action space of the wrapper.

    Args:
        env: The environment to wrap.

    Raises:
        ValueError: If the environment's action space is not a `Box`.
    """

    env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType, MaskType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    mask_func: Callable[[MaskType], MaskType]
    action_space: Box

    def __init__(
        self, env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType, MaskType]
    ):
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
        self.mask_func = lambda x: x
        self.action_space = action_space


class RescaleAction[StateType: AbstractEnvLikeState, ObsType, MaskType](
    AbstractPureTransformActionWrapper[
        Float[Array, " ..."],
        MaskType,
        StateType,
        Float[Array, " ..."],
        ObsType,
        MaskType,
    ]
):
    """
    Affine rescaling of a box action to a different range.

    Note:
        Only compatible with `Box` action spaces.

    Attributes:
        env: The environment to wrap.
        action_space: The action space of the wrapper.

    Args:
        env: The environment to wrap.

    Raises:
        ValueError: If the environment's action space is not a `Box`.
    """

    env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType, MaskType]
    func: Callable[[Float[Array, " ..."]], Float[Array, " ..."]]
    mask_func: Callable[[MaskType], MaskType]
    action_space: Box

    def __init__(
        self,
        env: AbstractEnvLike[StateType, Float[Array, " ..."], ObsType, MaskType],
        min: Float[Array, " ..."] = jnp.array(-1.0),
        max: Float[Array, " ..."] = jnp.array(1.0),
    ):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                "RescaleAction only supports `Box` action spaces"
                f" not {type(env.action_space)}"
            )

        action_space, _, rescale = rescale_box(env.action_space, min, max)

        self.env = env
        self.func = rescale
        self.mask_func = lambda x: x
        self.action_space = action_space
