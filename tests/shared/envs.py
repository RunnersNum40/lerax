from __future__ import annotations

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int

from oryx.env import AbstractEnv
from oryx.spaces import Box


class EchoEnv(AbstractEnv[Float[Array, " n"], Float[Array, " n"]]):
    """
    Echoes the action back as the observation, never terminates.
    """

    state_index: eqx.nn.StateIndex[None]
    _action_space: Box
    _observation_space: Box

    def __init__(
        self,
        action_space: Box = Box(-jnp.inf, jnp.inf),
        observation_space: Box = Box(-jnp.inf, jnp.inf),
    ):
        self._action_space = action_space
        self._observation_space = observation_space
        self.state_index = eqx.nn.StateIndex(None)

    def reset(self, state, *, key):
        return state, self.observation_space.sample(key), {}

    def step(self, state, action, *, key):
        return (
            state,
            action,
            jnp.asarray(0.0),
            jnp.asarray(False),
            jnp.asarray(False),
            {},
        )

    def render(self, state): ...
    def close(self): ...

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space


class FiniteEpisodeEnv(AbstractEnv[Float[Array, ""], Float[Array, ""]]):
    """
    Counts steps and terminates after ``done_at`` steps.
    """

    state_index: eqx.nn.StateIndex[Int[Array, ""]]
    done_at: int

    def __init__(self, *, key, done_at: int = 3):
        self.state_index = eqx.nn.StateIndex(jnp.array(0))
        self.done_at = done_at

    def reset(self, state, *, key):
        state = state.set(self.state_index, jnp.array(0))
        return state, jnp.asarray(0.0), {}

    def step(self, state, action, *, key):
        step_no = state.get(self.state_index) + 1
        state = state.set(self.state_index, step_no)

        termination = step_no == self.done_at
        return (
            state,
            step_no,
            jnp.asarray(1.0),
            termination,
            jnp.asarray(False),
            {},
        )

    def render(self, state): ...
    def close(self): ...

    @property
    def action_space(self):
        return Box(-jnp.inf, jnp.inf)

    @property
    def observation_space(self):
        return Box(-jnp.inf, jnp.inf)


class PassThroughEnv(AbstractEnv[Float[Array, ""], Float[Array, ""]]):
    """
    Deterministic scalar environment that echoes the action as the reward and
    always returns observation = 0.0.
    """

    state_index: eqx.nn.StateIndex[None]

    def __init__(self):
        self.state_index = eqx.nn.StateIndex(None)

    def reset(self, state, *, key):
        return state, jnp.asarray(0.0), {}

    def step(self, state, action, *, key):
        return (
            state,
            jnp.asarray(0.0),
            jnp.asarray(action),
            jnp.asarray(False),
            jnp.asarray(False),
            {},
        )

    def render(self, state): ...
    def close(self): ...

    @property
    def action_space(self):
        return Box(-jnp.inf, jnp.inf)

    @property
    def observation_space(self):
        return Box(-jnp.inf, jnp.inf)


class DummyEnv(AbstractEnv[Float[Array, ""], Float[Array, ""]]):
    """
    Scalar environment for unit tests.

    *Observation* =`state_value`
    *Reward* =`-abs(action - state_value)`
    Never terminates/truncates.
    """

    state_index: eqx.nn.StateIndex[Float[Array, ""]]

    def __init__(self, *, key):
        self.state_index = eqx.nn.StateIndex(jr.uniform(key))

    def reset(self, state, *, key):
        new_val = jr.uniform(key)
        state = state.set(self.state_index, new_val)
        return state, new_val, {}

    def step(self, state, action: Float[Array, ""], *, key):
        new_val = jr.uniform(key)
        reward = -jnp.abs(action - new_val)
        state = state.set(self.state_index, new_val)
        done = jnp.asarray(False)
        trunc = jnp.asarray(False)
        return state, new_val, reward, done, trunc, {}

    def render(self, state):
        pass

    def close(self):
        pass

    @property
    def action_space(self) -> Box:
        return Box(-jnp.inf, jnp.inf, shape=())

    @property
    def observation_space(self) -> Box:
        return Box(-jnp.inf, jnp.inf, shape=())
