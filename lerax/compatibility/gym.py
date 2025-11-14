from __future__ import annotations

from typing import ClassVar, Literal

import gymnasium as gym
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jax.debug import callback as debug_callback
from jax.experimental import io_callback
from jaxtyping import Array, Bool, Float, Key

from lerax.env import AbstractEnv, AbstractEnvState
from lerax.render import AbstractRenderer
from lerax.space import AbstractSpace, Box, Dict, Discrete, Tuple


def gym_space_to_lerax_space(space: gym.Space) -> AbstractSpace:
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape)
    elif isinstance(space, gym.spaces.Dict):
        return Dict({k: gym_space_to_lerax_space(s) for k, s in space.spaces.items()})
    elif isinstance(space, gym.spaces.Tuple):
        return Tuple(tuple(gym_space_to_lerax_space(s) for s in space.spaces))
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def lerax_to_gym_space(space: AbstractSpace) -> gym.Space:
    if isinstance(space, Discrete):
        return gym.spaces.Discrete(int(space.n), start=int(space.start))
    elif isinstance(space, Box):
        return gym.spaces.Box(
            low=np.asarray(space.low),
            high=np.asarray(space.high),
        )
    elif isinstance(space, Dict):
        return gym.spaces.Dict(
            {k: lerax_to_gym_space(s) for k, s in space.spaces.items()}
        )
    elif isinstance(space, Tuple):
        return gym.spaces.Tuple(tuple(lerax_to_gym_space(s) for s in space.spaces))
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def jax_to_numpy(x):
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    return x


def to_numpy_tree(x):
    return jax.tree.map(jax_to_numpy, x)


class GymEnvState(AbstractEnvState):
    observation: Array
    reward: Float[Array, ""]
    terminal: Bool[Array, ""]
    truncated: Bool[Array, ""]


class GymToLeraxEnv(AbstractEnv[GymEnvState, Array, Array]):
    """
    Wrapper of a Gymnasium environment to make it compatible with Lerax.

    Uses jax's io_callback to wrap the env's reset and step functions.
    In general, this will be slower than a native JAX environment and prevents
    vmapped rollout. Also removes the info dict returned by Gymnasium envs since
    the shape cannot be known ahead of time.
    """

    name: ClassVar[str] = "GymnasiumEnv"

    action_space: AbstractSpace
    observation_space: AbstractSpace

    env: gym.Env

    renderer: None = None

    def __init__(self, env: gym.Env):
        self.env = env
        self.action_space = gym_space_to_lerax_space(env.action_space)
        self.observation_space = gym_space_to_lerax_space(env.observation_space)

    def _reset(self, *args, **kwargs):
        if "seed" in kwargs:
            kwargs["seed"] = int(kwargs["seed"])

        obs, _ = self.env.reset(*args, **kwargs)
        return jnp.asarray(obs)

    def initial(self, *args, key: Key, **kwargs) -> GymEnvState:
        # TODO: Determine if we want to pass a seed or not
        # I think it's a nice perk to increase reproducibility but it might
        # be unexpected for some users
        if "seed" not in kwargs:
            kwargs["seed"] = jr.randint(key, (), 0, jnp.iinfo(jnp.int32).max)

        observation = io_callback(self._reset, self.observation_space.canonical())
        return GymEnvState(
            observation=observation,
            reward=jnp.array(0.0, dtype=float),
            terminal=jnp.array(False, dtype=bool),
            truncated=jnp.array(False, dtype=bool),
        )

    def _step(self, action: Array):
        observation, reward, terminated, truncated, _ = self.env.step(
            np.asarray(action)
        )

        return (
            jnp.asarray(observation),
            jnp.asarray(reward),
            jnp.asarray(terminated),
            jnp.asarray(truncated),
        )

    def transition(self, state: GymEnvState, action: Array, *, key: Key) -> GymEnvState:
        observation, reward, terminated, truncated = io_callback(
            self._step,
            (
                self.observation_space.canonical(),
                jnp.array(0.0, dtype=float),
                jnp.array(False, dtype=bool),
                jnp.array(False, dtype=bool),
            ),
            action,
        )

        return GymEnvState(
            observation=observation,
            reward=reward,
            terminal=terminated,
            truncated=truncated,
        )

    def observation(self, state: GymEnvState, *, key: Key) -> Array:
        return state.observation

    def reward(
        self, state: GymEnvState, action: Array, next_state: GymEnvState, *, key: Key
    ) -> Float[Array, ""]:
        return next_state.reward

    def terminal(self, state: GymEnvState, *, key: Key) -> Bool[Array, ""]:
        return state.terminal

    def truncate(self, state: GymEnvState) -> Bool[Array, ""]:
        return state.truncated

    def state_info(self, state: GymEnvState) -> dict:
        return {}

    def transition_info(
        self, state: GymEnvState, action: Array, next_state: GymEnvState
    ) -> dict:
        return {}

    def render(self, state: GymEnvState, renderer: AbstractRenderer):
        raise NotImplementedError("Rendering not implemented for GymToLeraxEnv")

    def default_renderer(self) -> AbstractRenderer:
        raise NotImplementedError("Default renderer not implemented for GymToLeraxEnv.")

    def close(self):
        debug_callback(self.env.close, ordered=True)


class LeraxToGymEnv[StateType: AbstractEnvState](gym.Env):
    """
    Wrapper of an Lerax environment to make it compatible with Gymnasium.

    Executes the Lerax env directly (Python side). Keeps an internal eqx state and PRNG.
    """

    metadata: dict = {"render_modes": ["human"]}

    action_space: gym.Space
    observation_space: gym.Space

    render_mode: str | None = None

    env: AbstractEnv[StateType, Array, Array]
    state: StateType
    key: Key

    def __init__(
        self,
        env: AbstractEnv[StateType, Array, Array],
        render_mode: Literal["human"] | None = None,
    ):
        self.key = jr.key(0)

        self.env = env

        self.action_space = lerax_to_gym_space(env.action_space)
        self.observation_space = lerax_to_gym_space(env.observation_space)

        self.render_mode = render_mode
        # TODO: Actually handle rendering

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.key = jr.key(int(seed))

        self.key, reset_key = jr.split(self.key)
        self.state, obs, info = self.env.reset(key=reset_key)
        return jax_to_numpy(obs), to_numpy_tree(info)

    def step(self, action):
        self.key, step_key = jr.split(self.key)
        self.state, obs, rew, term, trunc, info = self.env.step(
            self.state, jnp.asarray(action), key=step_key
        )

        return (
            jax_to_numpy(obs),
            float(jnp.asarray(rew)),
            bool(jnp.asarray(term)),
            bool(jnp.asarray(trunc)),
            to_numpy_tree(info),
        )

    def render(self):
        raise NotImplementedError("Rendering not implemented for LeraxToGymEnv")

    def close(self): ...
