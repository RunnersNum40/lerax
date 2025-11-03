from __future__ import annotations

import dataclasses
from functools import partial

import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Key, PyTree

from lerax.policy import AbstractPolicyState

from .base_buffer import AbstractBuffer


class RolloutBuffer[StateType: AbstractPolicyState, ActType, ObsType](AbstractBuffer):
    """
    RolloutBuffer used by on-policy algorithms.

    Designed for scans and JIT compilation.
    """

    observations: PyTree[ObsType]
    actions: PyTree[ActType]
    rewards: Float[Array, " *size"]
    terminations: Bool[Array, " *size"]
    truncations: Bool[Array, " *size"]
    log_probs: Float[Array, " *size"]
    values: Float[Array, " *size"]
    returns: Float[Array, " *size"]
    advantages: Float[Array, " *size"]
    states: StateType

    def __init__(
        self,
        observations: PyTree[ObsType],
        actions: PyTree[ActType],
        rewards: Float[ArrayLike, " *size"],
        terminations: Bool[ArrayLike, " *size"],
        truncations: Bool[ArrayLike, " *size"],
        log_probs: Float[ArrayLike, " *size"],
        values: Float[ArrayLike, " *size"],
        states: StateType,
        returns: Float[ArrayLike, " *size"] | None = None,
        advantages: Float[ArrayLike, " *size"] | None = None,
    ):
        """
        Initialize the RolloutBuffer with the given parameters.

        Returns and advantages can be provided, but if not, they will be filled with
        NaNs.
        """
        # TODO: Add type checks for observations, actions, etc
        # TODO: Add shape checks for rewards, terminations, truncations, log_probs,
        # values

        self.observations = observations
        self.actions = actions
        self.rewards = jnp.asarray(rewards)
        self.terminations = jnp.asarray(terminations)
        self.truncations = jnp.asarray(truncations)
        self.log_probs = jnp.asarray(log_probs)
        self.values = jnp.asarray(values)
        # Assume the state has been used already and clone to compensate
        self.states = states
        self.returns = (
            jnp.asarray(returns)
            if returns is not None
            else jnp.full_like(values, jnp.nan)
        )
        self.advantages = (
            jnp.asarray(advantages)
            if advantages is not None
            else jnp.full_like(values, jnp.nan)
        )

    def compute_returns_and_advantages(
        self,
        last_value: Float[ArrayLike, ""],
        done: Bool[ArrayLike, ""],
        gae_lambda: Float[ArrayLike, ""],
        gamma: Float[ArrayLike, ""],
    ) -> RolloutBuffer[StateType, ActType, ObsType]:
        """
        Compute returns and advantages for the rollout buffer using Generalized
        Advantage Estimation.

        Works under JIT compilation.
        """
        last_value = jnp.asarray(last_value)
        done = jnp.asarray(done)
        gamma = jnp.asarray(gamma)
        gae_lambda = jnp.asarray(gae_lambda)

        dones = jnp.logical_or(self.terminations, self.truncations)

        next_values = jnp.concatenate(
            [self.values[1:], jnp.array([last_value])], axis=0
        )

        next_non_terminal = jnp.concatenate(
            [1.0 - dones[1:], jnp.array([1.0 - done])], axis=0
        )

        deltas = self.rewards + gamma * next_values * next_non_terminal - self.values

        def scan_fn(
            advantage_carry: Float[Array, ""],
            x: tuple[Float[Array, ""], Float[Array, ""]],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            delta, next_non_terminal = x
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage_carry
            return advantage, advantage

        _, advantages = lax.scan(
            scan_fn, jnp.array(0.0), (jnp.flip(deltas), jnp.flip(next_non_terminal))
        )
        advantages = jnp.flip(advantages)
        returns = advantages + self.values

        return dataclasses.replace(self, advantages=advantages, returns=returns)

    def batches(
        self,
        batch_size: int,
        *,
        key: Key | None = None,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> RolloutBuffer[StateType, ActType, ObsType]:
        ndim = self.rewards.ndim

        if batch_axes is None:
            axes = tuple(range(ndim))
        elif isinstance(batch_axes, int):
            axes = (batch_axes,)
        else:
            axes = tuple(batch_axes)

        axes = tuple(a + ndim if a < 0 else a for a in axes)
        if len(set(axes)) != len(axes) or any(a < 0 or a >= ndim for a in axes):
            raise ValueError(f"Invalid batch_axes {batch_axes} for array ndim={ndim}.")

        def flatten_selected(x):
            moved = jnp.moveaxis(x, axes, tuple(range(len(axes))))
            lead = 1
            for i in range(len(axes)):
                lead *= moved.shape[i]

            return moved.reshape((lead,) + moved.shape[len(axes) :])

        flat_self = jax.tree.map(flatten_selected, self)

        total = flat_self.rewards.shape[0]
        indices = jnp.arange(total) if key is None else jr.permutation(key, total)

        if total % batch_size != 0:
            total_trim = total - (total % batch_size)
            indices = indices[:total_trim]

        indices = indices.reshape(-1, batch_size)

        return jax.tree.map(
            lambda x: jnp.take(x, indices, axis=0) if isinstance(x, jnp.ndarray) else x,
            flat_self,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rewards.shape
