from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Literal, Self, Sequence

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


class AbstractEnvLike[StateType: AbstractEnvLikeState, ActType, ObsType, MaskType](
    eqx.Module
):
    """Base class for RL environments or wrappers that behave like environments"""

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType, MaskType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType, Any]]

    @abstractmethod
    def initial(self, *, key: Key[Array, ""]) -> StateType:
        """
        Generate the initial state of the environment.

        Args:
            key: A JAX PRNG key for any stochasticity in the initial state.

        Returns:
            An initial environment state.
        """

    @abstractmethod
    def action_mask(self, state: StateType, *, key: Key[Array, ""]) -> MaskType | None:
        """
        Generate an action mask from the environment state.

        Args:
            state: The current environment state.
            key: A JAX PRNG key for any stochasticity in the action mask.

        Returns:
            A mask indicating valid and invalid actions for the environment state.
        """

    @abstractmethod
    def transition(
        self, state: StateType, action: ActType, *, key: Key[Array, ""]
    ) -> StateType:
        """
        Update the environment state given an action.

        Args:
            state: The current environment state.
            action: The action to take.
            key: A JAX PRNG key for any stochasticity in the transition.

        Returns:
            The next environment state.
        """

    @abstractmethod
    def observation(self, state: StateType, *, key: Key[Array, ""]) -> ObsType:
        """
        Generate an observation from the environment state.

        Args:
            state: The current environment state.
            key: A JAX PRNG key for any stochasticity in the observation.

        Returns:
            An observation corresponding to the environment state.
        """

    @abstractmethod
    def reward(
        self,
        state: StateType,
        action: ActType,
        next_state: StateType,
        *,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        """
        Generate a reward from the environment state transition.

        Args:
            state: The current environment state.
            action: The action taken.
            next_state: The next environment state.
            key: A JAX PRNG key for any stochasticity in the reward.

        Returns:
            A reward corresponding to the environment state transition.
        """

    @abstractmethod
    def terminal(self, state: StateType, *, key: Key[Array, ""]) -> Bool[Array, ""]:
        """
        Determine whether the environment state is terminal.

        Args:
            state: The current environment state.
            key: A JAX PRNG key for any stochasticity in the terminal condition.

        Returns:
            A boolean indicating whether the environment state is terminal.
        """

    @abstractmethod
    def truncate(self, state: StateType) -> Bool[Array, ""]:
        """
        Determine whether the environment state is truncated.

        Args:
            state: The current environment state.

        Returns:
            A boolean indicating whether the environment state is truncated.
        """

    @abstractmethod
    def state_info(self, state: StateType) -> dict:
        """
        Generate additional info from the environment state.

        In many cases, this can simply return an empty dictionary.

        Args:
            state: The current environment state.

        Returns:
            A dictionary of additional info from the environment state.
        """

    @abstractmethod
    def transition_info(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> dict:
        """
        Generate additional info from the environment state transition.

        In many cases, this can simply return an empty dictionary.

        Args:
            state: The current environment state.
            action: The action taken.
            next_state: The next environment state.

        Returns:
            A dictionary of additional info from the environment state transition.
        """

    @abstractmethod
    def default_renderer(self) -> AbstractRenderer:
        """
        Return the default renderer for the environment.

        Returns:
            An instance of AbstractRenderer for rendering the environment.
        """

    @abstractmethod
    def render(self, state: StateType, renderer: AbstractRenderer):
        """
        Render a frame from a state.

        Args:
            state: The environment state to render.
            renderer: The renderer to use for rendering.
        """

    def render_states(
        self,
        states: Sequence[StateType],
        renderer: AbstractRenderer | Literal["auto"] = "auto",
        dt: float = 0.0,
    ):
        """
        Render a sequence of frames from multiple states.

        Args:
            states: A sequence of environment states to render.
            renderer: The renderer to use for rendering. If "auto", uses the default renderer.
            dt: The time delay between rendering each frame, in seconds.
        """
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
        """
        Render multiple frames from stacked states.

        Stacked states are typically batched states stored in a pytree structure.

        Args:
            states: A pytree of stacked environment states to render.
            renderer: The renderer to use for rendering. If "auto", uses the default renderer.
            dt: The time delay between rendering each frame, in seconds.
        """
        self.render_states(unstack_pytree(states), renderer, dt)

    @property
    @abstractmethod
    def unwrapped(self) -> AbstractEnv:
        """
        Return the unwrapped environment.

        Returns:
            The unwrapped environment.
        """

    @eqx.filter_jit
    def reset(self, *, key: Key[Array, ""]) -> tuple[StateType, ObsType, dict]:
        """
        Wrap the functional logic into a Gym API reset method.

        Args:
            key: A JAX PRNG key for any stochasticity in the reset.

        Returns:
            A tuple of the initial state, initial observation, and additional info.
        """
        initial_key, observation_key = jr.split(key, 2)
        state = self.initial(key=initial_key)
        observation = self.observation(state, key=observation_key)
        info = self.state_info(state)
        return state, observation, info

    @eqx.filter_jit
    def step(
        self, state: StateType, action: ActType, *, key: Key[Array, ""]
    ) -> tuple[
        StateType, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        """
        Wrap the functional logic into a Gym API step method.

        Args:
            state: The current environment state.
            action: The action to take.
            key: A JAX PRNG key for any stochasticity in the step.

        Returns:
            A tuple of the next state, observation, reward, terminal flag, truncate flag, and additional info.
        """
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
    """Base class for RL environment states."""

    @property
    def unwrapped(self) -> AbstractEnvState:
        """Return the unwrapped environment state"""
        return self


class AbstractEnv[StateType: AbstractEnvState, ActType, ObsType, MaskType](
    AbstractEnvLike[StateType, ActType, ObsType, MaskType]
):
    """
    Base class for RL environments.

    Attributes:
        name: The name of the environment
        action_space: The action space of the environment
        observation_space: The observation space of the environment
    """

    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[AbstractSpace[ActType, MaskType]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType, Any]]

    @property
    def unwrapped(self) -> Self:
        """Return the unwrapped environment"""
        return self
