from __future__ import annotations

from typing import Any

try:
    from flax import struct
    from gymnax.environments import environment as gym
    from gymnax.environments import spaces as gym_spaces
except ModuleNotFoundError as e:
    raise ImportError(
        "Gymnax compatibility requires toptional dependencies. "
        "Install with `pip install lerax[compatibility]` or install "
        "`gymnax` and `flax` manually."
    ) from e
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.env import AbstractEnv, AbstractEnvState
from lerax.render import AbstractRenderer
from lerax.space import AbstractSpace, Box, Dict, Discrete, Tuple


def gymnax_space_to_lerax_space(space: gym_spaces.Space) -> AbstractSpace:
    """
    Returns a Lerax space corresponding to the given Gymnax space.

    Args:
        space: Gymnax space to convert.

    Returns:
        The corresponding Lerax space.
    """
    if isinstance(space, gym_spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym_spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape)
    elif isinstance(space, gym_spaces.Dict):
        return Dict(
            {k: gymnax_space_to_lerax_space(v) for k, v in space.spaces.items()}
        )
    elif isinstance(space, gym_spaces.Tuple):
        return Tuple(tuple(gymnax_space_to_lerax_space(s) for s in space.spaces))
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def lerax_to_gymnax_space(space: AbstractSpace) -> gym_spaces.Space:
    """
    Returns a Gymnax space corresponding to the given Lerax space.

    Args:
        space: Lerax space to convert.

    Returns:
        The corresponding Gymnax space.
    """
    if isinstance(space, Discrete):
        return gym_spaces.Discrete(int(space.n))
    elif isinstance(space, Box):
        return gym_spaces.Box(low=space.low, high=space.high, shape=space.shape)
    elif isinstance(space, Dict):
        return gym_spaces.Dict(
            {k: lerax_to_gymnax_space(v) for k, v in space.spaces.items()}
        )
    elif isinstance(space, Tuple):
        return gym_spaces.Tuple(tuple(lerax_to_gymnax_space(s) for s in space.spaces))
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


class GymnaxEnvState(AbstractEnvState):
    env_state: gym.EnvState
    observation: Array
    reward: Float[Array, ""]
    terminal: Bool[Array, ""]
    transition_info: dict


class GymnaxToLeraxEnv(AbstractEnv[GymnaxEnvState, Array, Array, None]):
    """
    Wrapper of a Gymnax environment to make it compatible with Lerax.

    Note:
        For the sake of simplicity, truncation is not handled and always set to False.
        To keep the API consistent, info returned by step is always an empty dict.

    Attributes:
        action_space: Action space of the environment.
        observation_space: Observation space of the environment.
        env: Gymnax environment being wrapped.
        params: Parameters for the Gymnax environment.

    Args:
        env: Gymnax environment to wrap.
        params: Parameters for the Gymnax environment.
    """

    name: str

    action_space: AbstractSpace
    observation_space: AbstractSpace

    env: gym.Environment
    params: gym.EnvParams

    def __init__(self, env: gym.Environment, params: gym.EnvParams):
        self.name = f"GymnaxToLeraxEnv({env.name})"
        self.env = env
        self.params = params

        self.action_space = gymnax_space_to_lerax_space(env.action_space(params))
        self.observation_space = gymnax_space_to_lerax_space(
            env.observation_space(params)
        )

    def initial(self, *, key: Key[Array, ""]) -> GymnaxEnvState:
        observation, env_state = self.env.reset_env(key, self.params)
        return GymnaxEnvState(
            env_state=env_state,
            observation=observation,
            reward=jnp.array(0.0, dtype=float),
            terminal=jnp.array(False, dtype=bool),
            transition_info={},
        )

    def action_mask(self, state: GymnaxEnvState, *, key: Key[Array, ""]) -> None:
        return None

    def transition(
        self, state: GymnaxEnvState, action: Array, *, key: Key[Array, ""]
    ) -> GymnaxEnvState:
        observation, env_state, reward, done, _ = self.env.step_env(
            key, state.env_state, action, self.params
        )
        return GymnaxEnvState(
            env_state=env_state,
            observation=observation,
            reward=reward,
            terminal=done,
            transition_info={},
        )

    def observation(self, state: GymnaxEnvState, *, key: Key[Array, ""]) -> Array:
        return state.observation

    def reward(
        self,
        state: GymnaxEnvState,
        action: Array,
        next_state: GymnaxEnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        return next_state.reward

    def terminal(
        self, state: GymnaxEnvState, *, key: Key[Array, ""]
    ) -> Bool[Array, ""]:
        return state.terminal

    def truncate(self, state: GymnaxEnvState) -> Bool[Array, ""]:
        return jnp.array(False, dtype=bool)

    def state_info(self, state: GymnaxEnvState) -> dict:
        return {}

    def transition_info(
        self, state: GymnaxEnvState, action: Array, next_state: GymnaxEnvState
    ) -> dict:
        return next_state.transition_info

    def render(self, state: GymnaxEnvState, renderer: AbstractRenderer):
        raise NotImplementedError("Rendering not implemented for GymnaxToLeraxEnv.")

    def default_renderer(self) -> AbstractRenderer:
        raise NotImplementedError(
            "Default renderer not implemented for GymnaxToLeraxEnv."
        )


@struct.dataclass
class LeraxEnvParams(gym.EnvParams):
    pass


@struct.dataclass
class LeraxEnvState[StateType: AbstractEnvState](gym.EnvState):
    env_state: StateType
    time: Int[Array, ""]


class LeraxToGymnaxEnv[StateType: AbstractEnvState](
    gym.Environment[LeraxEnvState[StateType], LeraxEnvParams]
):
    """
    Wrapper of an Lerax environment to make it compatible with Gymnax.

    Note:
        Since Gymnax does not have a truncation concept, truncation and
        termination are combined into a single "done" signal.

    Attributes:
        env: Lerax environment being wrapped.
        state: Current state of the environment.

    Args:
        env: Lerax environment to wrap.
    """

    env: AbstractEnv[StateType, Array, Array, Any]
    state: StateType

    def __init__(self, env: AbstractEnv[StateType, Array, Array, Any]):
        self.env = env

    def step_env(
        self,
        key: Key[Array, ""],
        state: LeraxEnvState[StateType],
        action: ArrayLike,
        params: LeraxEnvParams,
    ) -> tuple[
        Array, LeraxEnvState[StateType], Float[Array, ""], Bool[Array, ""], dict
    ]:
        env_state, observation, reward, termination, truncation, info = self.env.step(
            state.env_state, jnp.asarray(action), key=key
        )
        done = termination | truncation
        return (
            observation,
            LeraxEnvState(
                env_state=env_state,
                time=state.time + 1,
            ),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: Key[Array, ""], params: LeraxEnvParams
    ) -> tuple[Array, LeraxEnvState[StateType]]:
        initial_key, observation_key = jr.split(key, 2)
        env_state = self.env.initial(key=initial_key)
        observation = self.env.observation(env_state, key=observation_key)

        return observation, LeraxEnvState(env_state=env_state, time=jnp.array(0))

    def get_obs(  # pyright: ignore
        self,
        state: LeraxEnvState[StateType],
        key: Key[Array, ""],
        params: LeraxEnvParams,
    ) -> Array:
        return self.env.observation(state.env_state, key=key)

    def is_terminal(
        self, state: LeraxEnvState[StateType], params: LeraxEnvParams
    ) -> Bool[Array, ""]:
        return self.env.terminal(state.env_state, key=jr.key(-1))

    @property
    def name(self) -> str:
        return self.env.name

    @property
    def default_params(self) -> LeraxEnvParams:
        return LeraxEnvParams()

    def observation_space(self, params: LeraxEnvParams) -> gym_spaces.Space:
        return lerax_to_gymnax_space(self.env.observation_space)

    def action_space(self, params: LeraxEnvParams) -> gym_spaces.Space:
        return lerax_to_gymnax_space(self.env.action_space)

    def state_space(self, params: LeraxEnvParams) -> gym_spaces.Space:
        raise NotImplementedError("State space not implemented for LeraxToGymnaxEnv.")
