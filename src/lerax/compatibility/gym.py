from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

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
from lerax.space import (
    AbstractSpace,
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Tuple,
)


def gym_space_to_lerax_space(space: gym.Space) -> AbstractSpace:
    """
    Convert a Gymnasium space to a Lerax space.

    Args:
        space: Gymnasium space to convert.

    Returns:
        Corresponding Lerax space.
    """
    if isinstance(space, gym.spaces.Discrete):
        if not space.start == 0:
            raise NotImplementedError(
                "Gym Discrete space with non-zero start are not supported"
            )
        return Discrete(n=int(cast(int | np.integer[Any], space.n)))
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape)
    elif isinstance(space, gym.spaces.Dict):
        return Dict({k: gym_space_to_lerax_space(s) for k, s in space.spaces.items()})
    elif isinstance(space, gym.spaces.Tuple):
        return Tuple(tuple(gym_space_to_lerax_space(s) for s in space.spaces))
    elif isinstance(space, gym.spaces.MultiBinary):
        return MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return MultiDiscrete(nvec=tuple(int(n) for n in space.nvec))
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def lerax_to_gym_space(space: AbstractSpace) -> gym.Space:
    """
    Convert a Lerax space to a Gymnasium space.

    Args:
        space: Lerax space to convert.

    Returns:
        Corresponding Gymnasium space.
    """

    if isinstance(space, Discrete):
        return gym.spaces.Discrete(int(space.n))
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
    elif isinstance(space, MultiBinary):
        return gym.spaces.MultiBinary(
            n=int(space.n[0]) if len(space.n) == 1 else space.n
        )
    elif isinstance(space, MultiDiscrete):
        return gym.spaces.MultiDiscrete(nvec=list(space.nvec))
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


class GymToLeraxEnv(AbstractEnv[GymEnvState, Array, Array, None]):
    """
    Wrap a Gymnasium environment for Lerax.

    Note:
        `io_callback` makes `reset` and `step` slower than native JAX and prevents
        vmapped rollout. Gymnasium info is discarded because its shape is unknown.
        Call methods in order because state objects omit required internal state.

    Args:
        env: Gymnasium environment to wrap.

    Attributes:
        name: Name of the environment.
        action_space: Action space of the environment.
        observation_space: Observation space of the environment.
        env: The original Gymnasium environment.
    """

    name: ClassVar[str] = "GymnasiumEnv"

    action_space: AbstractSpace
    observation_space: AbstractSpace

    env: gym.Env

    def __init__(self, env: gym.Env):
        self.env = env
        self.action_space = gym_space_to_lerax_space(env.action_space)
        self.observation_space = gym_space_to_lerax_space(env.observation_space)

    def initial(self, *args: Any, key: Key[Array, ""], **kwargs: Any) -> GymEnvState:
        """
        Call the Gymnasium `reset` method.

        Note:
            The key generates a reproducible seed unless one is provided.

        Args:
            *args: Positional arguments to pass to `env.reset`.
            key: JAX PRNG key used to generate a seed when absent.
            **kwargs: Keyword arguments to pass to `env.reset`. If `seed` is provided,
                it overrides the generated seed.

        Returns:
            The initial environment state.
        """
        if "seed" in kwargs:
            kwargs = dict(kwargs)
            seed_value = kwargs.pop("seed")
            seed = jnp.asarray(seed_value, dtype=int)
        else:
            seed = jr.randint(key, (), 0, jnp.iinfo(jnp.int32).max)

        def reset_callback(seed_arr):
            seed_int = int(seed_arr)
            obs, _ = self.env.reset(*args, seed=seed_int, **kwargs)
            return jnp.asarray(obs)

        observation = io_callback(
            reset_callback,
            self.observation_space.canonical(),
            seed,
            ordered=True,
        )

        return GymEnvState(
            observation=observation,
            reward=jnp.array(0.0, dtype=float),
            terminal=jnp.array(False, dtype=bool),
            truncated=jnp.array(False, dtype=bool),
        )

    def action_mask(self, state: GymEnvState, *, key: Key[Array, ""]) -> None:
        return None

    def transition(
        self, state: GymEnvState, action: Array, *, key: Key[Array, ""]
    ) -> GymEnvState:
        """
        Call the Gymnasium `step` method through `io_callback`.

        The state is ignored, so call order matters.

        Args:
            state: Current environment state.
            action: Action to take.
            key: Unused.

        Returns:
            Next environment state.
        """

        def step_callback(action_arr):
            observation, reward, terminated, truncated, _ = self.env.step(
                np.asarray(action_arr)
            )
            return (
                jnp.asarray(observation),
                jnp.asarray(reward, dtype=float),
                jnp.asarray(terminated, dtype=bool),
                jnp.asarray(truncated, dtype=bool),
            )

        observation, reward, terminated, truncated = io_callback(
            step_callback,
            (
                self.observation_space.canonical(),
                jnp.array(0.0, dtype=float),
                jnp.array(False, dtype=bool),
                jnp.array(False, dtype=bool),
            ),
            action,
            ordered=True,
        )

        return GymEnvState(
            observation=observation,
            reward=reward,
            terminal=terminated,
            truncated=truncated,
        )

    def observation(self, state: GymEnvState, *, key: Key[Array, ""]) -> Array:
        """
        Return the stored Gymnasium observation.

        Args:
            state: Current environment state.

        Returns:
            Stored observation.
        """
        return state.observation

    def reward(
        self,
        state: GymEnvState,
        action: Array,
        next_state: GymEnvState,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        """
        Return the reward stored in the next state.

        Args:
            state: Current environment state.
            action: Action taken.
            next_state: Next environment state.

        Returns:
            Transition reward.
        """
        return next_state.reward

    def terminal(self, state: GymEnvState, *, key: Key[Array, ""]) -> Bool[Array, ""]:
        """
        Return the stored Gymnasium terminated flag.

        Args:
            state: Current environment state.

        Returns:
            Whether the state is terminal.
        """
        return state.terminal

    def truncate(self, state: GymEnvState) -> Bool[Array, ""]:
        """
        Return the stored Gymnasium truncated flag.

        Args:
            state: Current environment state.

        Returns:
            Whether the state is truncated.
        """
        return state.truncated

    def state_info(self, state: GymEnvState) -> dict:
        """
        Return empty info to keep JIT shapes stable.

        Args:
            state: Current environment state.

        Returns:
            Empty info.
        """
        return {}

    def transition_info(
        self, state: GymEnvState, action: Array, next_state: GymEnvState
    ) -> dict:
        """
        Return empty info to keep JIT shapes stable.

        Args:
            state: Current environment state.
            action: Action taken.
            next_state: Next environment state.

        Returns:
            Empty info.
        """
        return {}

    def render(self, state: GymEnvState, renderer: AbstractRenderer):
        """
        Reject unsupported Gymnasium rendering.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Rendering not implemented for GymToLeraxEnv")

    def default_renderer(self) -> AbstractRenderer:
        """
        Reject unsupported Gymnasium rendering.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Default renderer not implemented for GymToLeraxEnv.")

    def close(self):
        debug_callback(self.env.close, ordered=True)


class LeraxToGymEnv[StateType: AbstractEnvState](gym.Env):
    """
    Wrap a Lerax environment for Gymnasium.

    Run Lerax in Python with internal environment state and a PRNG key.

    Attributes:
        metadata: Metadata for the Gym environment.
        action_space: Action space of the environment.
        observation_space: Observation space of the environment.
        render_mode: Render mode for the environment.
        env: The Lerax environment to wrap.
        state: Current state of the Lerax environment.
        key: PRNG key for the environment.

    Args:
        env: Lerax environment to wrap.
        render_mode: Render mode for the environment.
    """

    metadata: dict = {"render_modes": ["human"]}

    action_space: gym.Space
    observation_space: gym.Space

    render_mode: str | None = None

    env: AbstractEnv[StateType, Array, Array, Any]
    state: StateType
    key: Key[Array, ""]

    def __init__(
        self,
        env: AbstractEnv[StateType, Array, Array, Any],
        render_mode: Literal["human"] | None = None,
    ):
        self.key = jr.key(0)

        self.env = env

        self.action_space = lerax_to_gym_space(env.action_space)
        self.observation_space = lerax_to_gym_space(env.observation_space)

        self.render_mode = render_mode

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
        """
        Reject unsupported rendering.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Rendering not implemented for LeraxToGymEnv")

    def close(self):
        """Implement the Gymnasium Env interface as a no-op."""
