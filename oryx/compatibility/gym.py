from __future__ import annotations

from typing import ClassVar, Literal

import equinox as eqx
import gymnasium as gym
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jax.debug import callback as debug_callback
from jax.experimental import io_callback
from jaxtyping import Array, Bool, Float, Key

from oryx.env import AbstractEnv
from oryx.space import AbstractSpace, Box, Discrete


def gym_space_to_oryx_space(space: gym.Space) -> Box | Discrete:
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape)
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def oryx_to_gym_space(space: AbstractSpace) -> gym.Space:
    if isinstance(space, Discrete):
        return gym.spaces.Discrete(int(space.n), start=int(space.start))
    elif isinstance(space, Box):
        return gym.spaces.Box(
            low=np.asarray(space.low),
            high=np.asarray(space.high),
        )
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def jax_to_numpy(x):
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    return x


def to_numpy_tree(x):
    return jax.tree.map(jax_to_numpy, x)


class GymnasiumEnv(AbstractEnv[Array, Array]):
    """
    Wrapper of a Gymnasium environment to make it compatible with Oryx.

    Uses jax's io_callback to wrap the env's reset and step functions.
    In general, this will be slower than a native JAX environment,
    Also removes the info dict returned by Gymnasium envs.
    """

    name: ClassVar[str] = "GymnasiumEnv"

    state_index: eqx.nn.StateIndex[None]
    action_space: Box | Discrete
    observation_space: Box | Discrete

    env: gym.Env = eqx.field(static=True)

    def __init__(self, env: gym.Env):
        self.env = env
        self.action_space = gym_space_to_oryx_space(env.action_space)
        self.observation_space = gym_space_to_oryx_space(env.observation_space)
        self.state_index = eqx.nn.StateIndex(None)

    def _reset(self, *args, **kwargs):
        # TODO: Log a warning that the info dict is discarded
        kwargs["seed"] = int(kwargs["seed"])
        obs, _ = self.env.reset(*args, **kwargs)
        obs = jnp.asarray(obs, dtype=self.observation_space.dtype)
        return obs, {}

    def reset(
        self, state: eqx.nn.State, *args, key: Key, **kwargs
    ) -> tuple[eqx.nn.State, Array, dict]:
        # TODO: Determine if we want to pass a seed or not
        # I think it's a nice perk to increase reproducibility but it might
        # be unexpected for some users
        kwargs["seed"] = kwargs.get(
            "seed",
            jr.randint(key, (), 0, jnp.iinfo(jnp.int32).max),
        )

        return state, *io_callback(
            self._reset,
            (self.observation_space.canonical(), {}),
            *args,
            ordered=True,
            **kwargs,
        )

    def _step(self, action: Array):
        obs, reward, terminated, truncated, _ = self.env.step(np.asarray(action))

        return (
            jnp.asarray(obs, dtype=self.observation_space.dtype),
            jnp.asarray(reward),
            jnp.asarray(terminated),
            jnp.asarray(truncated),
            {},
        )

    def step(
        self, state: eqx.nn.State, action: Array, *, key: Key
    ) -> tuple[
        eqx.nn.State, Array, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        return state, *io_callback(
            self._step,
            (
                self.observation_space.canonical(),
                jnp.array(0.0),
                jnp.array(False),
                jnp.array(False),
                {},
            ),
            action,
            ordered=True,
        )

    def render(self, state: eqx.nn.State):
        # TODO: Log a warning that render is bypassed
        pass

    def close(self):
        debug_callback(self.env.close, ordered=True)


class OryxEnv(gym.Env):
    """
    Wrapper of an Oryx environment to make it compatible with Gymnasium.

    Executes the Oryx env directly (Python side). Keeps an internal eqx state and PRNG.
    """

    metadata: dict = {"render_modes": ["human"]}

    action_space: gym.Space
    observation_space: gym.Space

    render_mode: str | None = None

    env: AbstractEnv
    state: eqx.nn.State
    key: Key

    def __init__(
        self,
        env: AbstractEnv,
        state: eqx.nn.State,
        render_mode: Literal["human"] | None = None,
    ):
        self.env = env
        self.state = state

        self.key = jr.key(0)

        self.action_space = oryx_to_gym_space(env.action_space)
        self.observation_space = oryx_to_gym_space(env.observation_space)

        self.render_mode = render_mode
        # TODO: Actually handle rendering

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.key = jr.key(int(seed))

        self.key, reset_key = jr.split(self.key)
        self.state, obs, info = self.env.reset(self.state, key=reset_key)
        return jax_to_numpy(obs), to_numpy_tree(info)

    def step(self, action):
        if isinstance(self.env.action_space, Discrete):
            act = jnp.asarray(action, dtype=self.env.action_space.dtype)
        elif isinstance(self.env.action_space, Box):
            act = jnp.asarray(action, dtype=self.env.action_space.dtype)
        else:
            raise NotImplementedError(
                f"Unsupported action space {type(self.env.action_space)}."
            )

        self.key, step_key = jr.split(self.key)
        self.state, obs, rew, term, trunc, info = self.env.step(
            self.state, act, key=step_key
        )

        return (
            jax_to_numpy(obs),
            float(jnp.asarray(rew)),
            bool(jnp.asarray(term)),
            bool(jnp.asarray(trunc)),
            to_numpy_tree(info),
        )

    def render(self):
        self.env.render(self.state)

    def close(self):
        self.env.close()
