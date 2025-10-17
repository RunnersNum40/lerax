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
        self, batch_size: int, *, key: Key | None = None
    ) -> RolloutBuffer[StateType, ActType, ObsType]:
        """
        Return rollout buffer with batches of the given size.

        The buffer is shuffled if a key is provided.

        This method reshapes the buffer into batches of the specified size. It is not
        the same as a list of butches, but rather a single buffer where each batch is a
        slice of the original data.
        """
        if key is None:
            indices = jnp.arange(self.shape[0])
        else:
            indices = jr.permutation(key, self.shape[0])

        if self.shape[0] % batch_size != 0:
            # TODO: Add warning if batch_size does not divide the number of samples
            indices = indices[: self.shape[0] - self.shape[0] % batch_size]

        indices = indices.reshape(-1, batch_size)

        return jax.tree.map(partial(jnp.take, indices=indices, axis=0), self)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rewards.shape
