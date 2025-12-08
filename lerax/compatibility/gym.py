from __future__ import annotations

from typing import ClassVar, Literal

try:
    import gymnasium as gym
except ModuleNotFoundError as e:
    raise ImportError(
        "Gymnasium compatibility requires optional dependencies. "
        "Install with `pip install lerax[compatibility]` or install "
        "`gymnasium` manually."
    ) from e
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
    Returns a Lerax space corresponding to the given Gymnasium space.

    Args:
        space: Gymnasium space to convert.

    Returns:
        The corresponding Lerax space.
    """
    if isinstance(space, gym.spaces.Discrete):
        if not space.start == 0:
            raise NotImplementedError(
                "Gym Discrete space with non-zero start are not supported"
            )
        return Discrete(n=int(space.n))
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape)
    elif isinstance(space, gym.spaces.Dict):
        return Dict({k: gym_space_to_lerax_space(s) for k, s in space.spaces.items()})
    elif isinstance(space, gym.spaces.Tuple):
        return Tuple(tuple(gym_space_to_lerax_space(s) for s in space.spaces))
    elif isinstance(space, gym.spaces.MultiBinary):
        return MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return MultiDiscrete(ns=tuple(int(n) for n in space.nvec))
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def lerax_to_gym_space(space: AbstractSpace) -> gym.Space:
    """
    Returns a Gymnasium space corresponding to the given Lerax space.

    Args:
        space: Lerax space to convert.

    Returns:
        The corresponding Gymnasium space.
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
        return gym.spaces.MultiDiscrete(nvec=list(space.ns))
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

    Note:
        Uses jax's `io_callback` to wrap the env's `reset` and `step` functions.
        In general, this will be slower than a native JAX environment and prevents
        vmapped rollout. Also removes the info dict returned by Gymnasium envs since
        the shape cannot be known ahead of time. Even more so than normal it is
        important to only call methods in order since the state objects do not
        contain all necessary information.

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

    def initial(self, *args, key: Key, **kwargs) -> GymEnvState:
        """
        Forwards to the Gymnasium reset.

        Note:
            A seed is generated if none is provided to increase reproducibility.

        Args:
            *args: Positional arguments to pass to `env.reset`.
            key: JAX PRNG key, used to generate a seed if none is provided.
            **kwargs: Keyword arguments to pass to `env.reset`. If "seed" is
                provided here, it overrides the key-generated seed.

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

    def transition(self, state: GymEnvState, action: Array, *, key: Key) -> GymEnvState:
        """
        Forwards to the Gymnasium step.

        In practice, this just calls the env's step function via io_callback.
        This means that the state is ignored and order of operations is important.

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

    def observation(self, state: GymEnvState, *, key: Key) -> Array:
        """
        Forwards to the Gymnasium observation.

        Args:
            state: Current environment state.

        Returns:
            Observation corresponding to the environment state.
        """
        return state.observation

    def reward(
        self, state: GymEnvState, action: Array, next_state: GymEnvState, *, key: Key
    ) -> Float[Array, ""]:
        """
        Forwards to the Gymnasium reward.

        In practice, this just reads the reward from the next_state.

        Args:
            state: Current environment state.
            action: Action taken.
            next_state: Next environment state.

        Returns:
            Reward obtained from the transition.
        """
        return next_state.reward

    def terminal(self, state: GymEnvState, *, key: Key) -> Bool[Array, ""]:
        """
        Forwards to the Gymnasium terminated flag.

        Args:
            state: Current environment state.

        Returns:
            Boolean indicating whether the state is terminal.
        """
        return state.terminal

    def truncate(self, state: GymEnvState) -> Bool[Array, ""]:
        """
        Forwards to the Gymnasium truncated flag.

        Args:
            state: Current environment state.

        Returns:
            Boolean indicating whether the state is truncated.
        """
        return state.truncated

    def state_info(self, state: GymEnvState) -> dict:
        """
        Empty info dict to ensure stable shapes for JIT compilation.

        Args:
            state: Current environment state.

        Returns:
            Empty info dict.
        """
        return {}

    def transition_info(
        self, state: GymEnvState, action: Array, next_state: GymEnvState
    ) -> dict:
        """
        Empty info dict to ensure stable shapes for JIT compilation.

        Args:
            state: Current environment state.
            action: Action taken.
            next_state: Next environment state.

        Returns:
            Empty info dict.
        """
        return {}

    def render(self, state: GymEnvState, renderer: AbstractRenderer):
        """
        Not supported for Gymnasium environments.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Rendering not implemented for GymToLeraxEnv")

    def default_renderer(self) -> AbstractRenderer:
        """
        Not supported for Gymnasium environments.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Default renderer not implemented for GymToLeraxEnv.")

    def close(self):
        debug_callback(self.env.close, ordered=True)


class LeraxToGymEnv[StateType: AbstractEnvState](gym.Env):
    """
    Wrapper of an Lerax environment to make it compatible with Gymnasium.

    Executes the Lerax env directly (Python side). Keeps an internal eqx state and PRNG.

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
        """
        Not supported yet.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Rendering not implemented for LeraxToGymEnv")

    def close(self):
        """
        Placeholder close method.

        Does nothing but completes the Gymnasium Env interface.
        """
