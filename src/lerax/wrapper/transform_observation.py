from functools import partial
from typing import Any, Callable

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Box

from .base_wrapper import AbstractWrapper, AbstractWrapperState
from .utils import rescale_box


class PureObservationState[StateType: AbstractEnvLikeState](
    AbstractWrapperState[StateType]
):
    env_state: StateType


class AbstractPureObservationWrapper[
    WrapperObsType,
    StateType: AbstractEnvLikeState,
    ActType,
    ObsType,
    MaskType,
](
    AbstractWrapper[
        PureObservationState[StateType],
        ActType,
        WrapperObsType,
        MaskType,
        StateType,
        ActType,
        ObsType,
        MaskType,
    ]
):
    """
    Apply a pure function to every observation that leaves the environment.
    """

    env: eqx.AbstractVar[AbstractEnvLike[StateType, ActType, ObsType, MaskType]]
    func: eqx.AbstractVar[Callable[[ObsType], WrapperObsType]]
    observation_space: eqx.AbstractVar[AbstractSpace[WrapperObsType, Any]]

    @property
    def action_space(self) -> AbstractSpace[ActType, MaskType]:
        return self.env.action_space

    def initial(self, *, key: Key[Array, ""]) -> PureObservationState[StateType]:
        return PureObservationState(self.env.initial(key=key))

    def action_mask(
        self, state: PureObservationState[StateType], *, key: Key[Array, ""]
    ) -> MaskType | None:
        return self.env.action_mask(state.env_state, key=key)

    def transition(
        self,
        state: PureObservationState[StateType],
        action: ActType,
        *,
        key: Key[Array, ""],
    ) -> PureObservationState[StateType]:
        return PureObservationState(
            self.env.transition(state.env_state, action, key=key)
        )

    def observation(
        self, state: PureObservationState[StateType], *, key: Key[Array, ""]
    ) -> WrapperObsType:
        return self.func(self.env.observation(state.env_state, key=key))

    def reward(
        self,
        state: PureObservationState[StateType],
        action: ActType,
        next_state: PureObservationState[StateType],
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        return self.env.reward(state.env_state, action, next_state.env_state, key=key)

    def terminal(
        self, state: PureObservationState[StateType], *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return self.env.terminal(state.env_state, key=key)

    def truncate(self, state: PureObservationState[StateType]) -> Bool[Array, ""]:
        return self.env.truncate(state.env_state)

    def state_info(self, state: PureObservationState[StateType]) -> dict:
        return self.env.state_info(state.env_state)

    def transition_info(
        self,
        state: PureObservationState[StateType],
        action: ActType,
        next_state: PureObservationState[StateType],
    ) -> dict:
        return self.env.transition_info(state.env_state, action, next_state.env_state)


class TransformObservation[
    WrapperObsType,
    StateType: AbstractEnvLikeState,
    ActType,
    ObsType,
    MaskType,
](
    AbstractPureObservationWrapper[
        WrapperObsType, StateType, ActType, ObsType, MaskType
    ]
):
    """
    Apply an arbitrary function to every observation that leaves the environment.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.
    """

    env: AbstractEnvLike[StateType, ActType, ObsType, MaskType]
    func: Callable[[ObsType], WrapperObsType]
    observation_space: AbstractSpace[WrapperObsType, Any]

    def __init__(
        self,
        env: AbstractEnvLike[StateType, ActType, ObsType, MaskType],
        func: Callable[[ObsType], WrapperObsType],
        observation_space: AbstractSpace[WrapperObsType, Any],
    ):
        self.env = env
        self.func = func
        self.observation_space = observation_space


class ClipObservation[StateType: AbstractEnvLikeState, MaskType](
    AbstractPureObservationWrapper[
        Float[Array, " ..."],
        StateType,
        Float[Array, " ..."],
        Float[Array, " ..."],
        MaskType,
    ],
):
    """
    Clips every observation to the environment's observation space.

    Note:
        Only works with `Box` observation spaces.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.

    Raises:
        ValueError: If the environment's observation space is not a `Box`.
    """

    env: AbstractEnvLike
    func: Callable
    observation_space: Box

    def __init__(self, env: AbstractEnvLike):
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                "ClipObservation only supports `Box` observation spaces"
                f" not {type(env.observation_space)}"
            )

        self.env = env
        self.func = partial(
            jnp.clip,
            min=env.observation_space.low,
            max=env.observation_space.high,
        )
        self.observation_space = env.observation_space


class RescaleObservation[StateType: AbstractEnvLikeState, MaskType](
    AbstractPureObservationWrapper[
        Float[Array, " ..."],
        StateType,
        Float[Array, " ..."],
        Float[Array, " ..."],
        MaskType,
    ],
):
    """
    Affine rescaling of the box observation space to a specified range.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.
        min: The minimum value of the rescaled observation space.
        max: The maximum value of the rescaled observation space.

    Raises:
        ValueError: If the environment's observation space is not a `Box`.
    """

    env: AbstractEnvLike
    func: Callable
    observation_space: Box

    def __init__(
        self,
        env: AbstractEnvLike,
        min: Float[Array, " ..."] = jnp.array(-1.0),
        max: Float[Array, " ..."] = jnp.array(1.0),
    ):
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                "RescaleObservation only supports `Box` observation spaces"
                f" not {type(env.action_space)}"
            )

        new_box, forward, _ = rescale_box(env.observation_space, min, max)

        self.env = env
        self.func = forward
        self.observation_space = new_box


class FlattenObservation[StateType: AbstractEnvLikeState, ObsType, MaskType](
    AbstractPureObservationWrapper[
        Float[Array, " flat"], StateType, Float[Array, " ..."], ObsType, MaskType
    ]
):
    """
    Flatten the observation space into a 1-D array.

    Attributes:
        env: The environment to wrap.
        observation_space: The observation space of the wrapper.

    Args:
        env: The environment to wrap.
    """

    env: AbstractEnvLike
    func: Callable
    observation_space: Box

    def __init__(self, env: AbstractEnvLike):
        self.env = env
        self.func = self.env.observation_space.flatten_sample
        self.observation_space = Box(
            -jnp.inf,
            jnp.inf,
            shape=(int(jnp.asarray(self.env.observation_space.flat_size)),),
        )
