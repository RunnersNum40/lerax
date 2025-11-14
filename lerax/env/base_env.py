from __future__ import annotations

import time
from abc import abstractmethod
from typing import Literal, Sequence

import equinox as eqx
from jax import lax
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from lerax.render import AbstractRenderer
from lerax.space import AbstractSpace
from lerax.utils import unstack_pytree


class AbstractEnvLikeState(eqx.Module):

    @property
    @abstractmethod
    def unwrapped(self) -> AbstractEnvState:
        """Return the unwrapped environment state"""


class AbstractEnvLike[StateType: AbstractEnvLikeState, ActType, ObsType](eqx.Module):
    """Base class for RL environments or wrappers that behave like environments"""

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @abstractmethod
    def initial(self, *, key: Key) -> StateType:
        """Generate the initial state of the environment"""

    @abstractmethod
    def transition(self, state: StateType, action: ActType, *, key: Key) -> StateType:
        """Update the environment state given an action"""

    @abstractmethod
    def observation(self, state: StateType, *, key: Key) -> ObsType:
        """Generate an observation from the environment state"""

    @abstractmethod
    def reward(
        self, state: StateType, action: ActType, next_state: StateType, *, key: Key
    ) -> Float[Array, ""]:
        """Generate a reward from the environment state transition"""

    @abstractmethod
    def terminal(self, state: StateType, *, key: Key) -> Bool[Array, ""]:
        """Determine whether the environment state is terminal"""

    @abstractmethod
    def truncate(self, state: StateType) -> Bool[Array, ""]:
        """Determine whether the environment state is truncated"""

    @abstractmethod
    def state_info(self, state: StateType) -> dict:
        """Generate additional info from the environment state"""

    @abstractmethod
    def transition_info(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> dict:
        """Generate additional info from the environment state transition"""

    @abstractmethod
    def default_renderer(self) -> AbstractRenderer:
        """Return the default renderer for the environment"""

    @abstractmethod
    def render(self, state: StateType, renderer: AbstractRenderer):
        """Render a frame from a state"""

    def render_states(
        self,
        states: Sequence[StateType],
        renderer: AbstractRenderer | Literal["auto"] = "auto",
        dt: float = 0.0,
    ):
        renderer = self.default_renderer() if renderer == "auto" else renderer
        renderer.open()
        for state in states:
            self.render(state, renderer)
            time.sleep(dt)
        renderer.close()

    def render_stacked(
        self,
        states: StateType,
        renderer: AbstractRenderer | Literal["auto"] = "auto",
        dt: float = 0.0,
    ):
        """Render multiple frames from stacked states"""
        self.render_states(unstack_pytree(states), renderer, dt)

    @property
    @abstractmethod
    def unwrapped(self) -> AbstractEnv:
        """Return the unwrapped environment"""

    @eqx.filter_jit
    def reset(self, *, key: Key) -> tuple[StateType, ObsType, dict]:
        """Wrap the functional logic into a Gym-like reset method"""
        initial_key, observation_key = jr.split(key, 2)
        state = self.initial(key=initial_key)
        observation = self.observation(state, key=observation_key)
        info = self.state_info(state)
        return state, observation, info

    @eqx.filter_jit
    def step(
        self, state: StateType, action: ActType, *, key: Key
    ) -> tuple[
        StateType, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        """Wrap the functional logic into a Gym-like step method"""
        transition_key, reward_key, terminal_key, reset_key = jr.split(key, 4)

        next_state = self.transition(state, action, key=transition_key)
        reward = self.reward(state, action, next_state, key=reward_key)
        terminal = self.terminal(next_state, key=terminal_key)
        truncate = self.truncate(next_state)
        info = self.transition_info(state, action, next_state)

        state = lax.cond(
            terminal | truncate, lambda: self.initial(key=reset_key), lambda: next_state
        )
        observation = self.observation(state, key=key)

        return state, observation, reward, terminal, truncate, info


class AbstractEnvState(AbstractEnvLikeState):
    @property
    def unwrapped(self) -> AbstractEnvState:
        """Return the unwrapped environment state"""
        return self


class AbstractEnv[StateType: AbstractEnvState, ActType, ObsType](
    AbstractEnvLike[StateType, ActType, ObsType]
):
    """Base class for RL environments"""

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType]]

    @property
    def unwrapped(self) -> AbstractEnv[StateType, ActType, ObsType]:
        """Return the unwrapped environment"""
        return self
