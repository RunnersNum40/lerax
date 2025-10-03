from __future__ import annotations

from typing import ClassVar

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from oryx.space import Box

from .base_env import AbstractEnv


class ContinuousMountainCar(AbstractEnv[Float[Array, ""], Float[Array, "2"]]):
    name: ClassVar[str] = "ContinuousMountainCar"

    state_index: eqx.nn.StateIndex[Float[Array, "2"]]
    action_space: Box
    observation_space: Box

    min_action: float
    max_action: float
    min_position: float
    max_position: float
    max_speed: float
    goal_position: float
    goal_velocity: float

    power: float

    low: Float[Array, "2"]
    high: Float[Array, "2"]

    def __init__(self, goal_velocity: float = 0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low = jnp.array([self.min_position, -self.max_speed])
        self.high = jnp.array([self.max_position, self.max_speed])

        self.action_space = Box(self.min_action, self.max_action)
        self.observation_space = Box(self.low, self.high)

        self.state_index = eqx.nn.StateIndex(jnp.zeros(2))

    def reset(
        self, state: eqx.nn.State, *, key: Key, low: float = -0.6, high: float = -0.4
    ) -> tuple[eqx.nn.State, Float[Array, "2"], dict]:
        position = jr.uniform(key, minval=low, maxval=high)
        velocity = 0.0

        state_vals = jnp.asarray([position, velocity])
        state = state.set(self.state_index, state_vals)

        return state, state_vals, {}

    def step(self, state: eqx.nn.State, action: Float[Array, ""], *, key: Key) -> tuple[
        eqx.nn.State,
        Float[Array, "2"],
        Float[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        dict,
    ]:
        position, velocity = state.get(self.state_index)
        force = jnp.clip(action, self.min_action, self.max_action)

        velocity += self.power * force - jnp.cos(3 * position) * 0.0025
        velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)

        position += velocity
        position = jnp.clip(position, self.min_position, self.max_position)

        velocity = lax.cond(
            jnp.logical_and(position == self.min_position, velocity < 0),
            lambda: jnp.array(0.0),
            lambda: velocity,
        )

        terminated = jnp.logical_and(
            position >= self.goal_position, velocity >= self.goal_velocity
        )

        reward = lax.cond(
            terminated,
            lambda: 0.0,
            lambda: 100.0,
        )

        state_vals = jnp.asarray([position, velocity])
        state = state.set(self.state_index, state_vals)

        return state, state_vals, reward, terminated, jnp.array(False), {}

    def render(self, state: eqx.nn.State):
        raise NotImplementedError

    def close(self): ...
