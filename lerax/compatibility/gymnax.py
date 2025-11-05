from __future__ import annotations

from flax import struct
from gymnax.environments import environment as gym
from gymnax.environments import spaces as gym_spaces
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.env import AbstractEnv, AbstractEnvState
from lerax.space import AbstractSpace, Box, Dict, Discrete, Tuple


def gymnax_space_to_lerax_space(space: gym_spaces.Space) -> AbstractSpace:
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


class GymnaxToLeraxEnv(AbstractEnv[GymnaxEnvState, Array, Array]):
    """
    Wrapper of a Gymnax environment to make it compatible with Lerax.

    For the sake of simplicity, truncation is not handled and always set to False.
    To keep the API consistent, info returned by step is always an empty dict.
    """

    action_space: AbstractSpace
    observation_space: AbstractSpace

    env: gym.Environment
    params: gym.EnvParams

    renderer: None = None

    def __init__(self, env: gym.Environment, params: gym.EnvParams):
        self.env = env
        self.params = params

        self.action_space = gymnax_space_to_lerax_space(env.action_space(params))
        self.observation_space = gymnax_space_to_lerax_space(
            env.observation_space(params)
        )

    def reset(self, *, key: Key) -> tuple[GymnaxEnvState, Array, dict]:
        obs, env_state = self.env.reset(key=key, params=self.params)
        return GymnaxEnvState(env_state=env_state), obs, {}

    def step(
        self, state: GymnaxEnvState, action: Array, *, key: Key
    ) -> tuple[
        GymnaxEnvState, Array, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action, self.params
        )
        return (
            GymnaxEnvState(env_state=env_state),
            obs,
            reward,
            done,
            jnp.array(False),
            {},  # info must be ignored to keep the shape consistent with reset
        )

    def render(self, state: GymnaxEnvState) -> None:
        raise NotImplementedError("Rendering not implemented for GymnaxToLeraxEnv.")

    def close(self): ...

    @property
    def name(self) -> str:
        return f"GymnaxToLeraxEnv({self.env.name})"


@struct.dataclass
class LeraxEnvParams(gym.EnvParams):
    pass


@struct.dataclass
class LeraxEnvState[StateType: AbstractEnvState](gym.EnvState):
    env_state: StateType
    terminal: Bool[Array, ""]
    observation: Array
    time: Int[Array, ""]


class LeraxToGymnaxEnv[StateType: AbstractEnvState](
    gym.Environment[LeraxEnvState[StateType], LeraxEnvParams]
):
    """
    Wrapper of an Lerax environment to make it compatible with Gymnax.
    """

    env: AbstractEnv[StateType, Array, Array]
    state: StateType
    key: Key

    def __init__(self, env: AbstractEnv[StateType, Array, Array]):
        self.key = jr.key(0)
        self.env = env

    def step_env(
        self,
        key: Key,
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
                observation=observation,
                terminal=done,
                time=state.time + 1,
            ),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: Key, params: LeraxEnvParams
    ) -> tuple[Array, LeraxEnvState[StateType]]:
        env_state, observation, info = self.env.reset(key=key)
        return observation, LeraxEnvState(
            env_state=env_state,
            observation=observation,
            terminal=jnp.array(False),
            time=jnp.array(0),
        )

    def get_obs(
        self,
        state: LeraxEnvState[StateType],
        params: LeraxEnvParams | None = None,
        key: Key | None = None,
    ) -> Array:
        return state.observation

    def is_terminal(
        self, state: LeraxEnvState[StateType], params: LeraxEnvParams
    ) -> Bool[Array, ""]:
        return state.terminal

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
