from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Int, Key
from tensorboardX import SummaryWriter

from oryx.buffers import RolloutBuffer
from oryx.env import AbstractEnvLike
from oryx.policies import AbstractActorCriticPolicy
from oryx.utils import clone_state, debug_with_numpy_wrapper, filter_scan

from .base_algorithm import AbstractAlgorithm
from .utils import create_progress_bar


class StepCarry[ObsType](eqx.Module):
    """Carry for the step function."""

    step_count: Int[Array, ""]
    last_obs: ObsType
    last_termination: Bool[Array, ""]
    last_truncation: Bool[Array, ""]


class IterationCarry[ActType, ObsType](eqx.Module):
    """Carry for the epoch function."""

    step_carry: StepCarry[ObsType]
    policy: AbstractActorCriticPolicy[Float, ActType, ObsType]


class AbstractOnPolicyAlgorithm[ActType, ObsType](AbstractAlgorithm[ActType, ObsType]):
    """Base class for on policy algorithms."""

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]
    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]
    policy: eqx.AbstractVar[AbstractActorCriticPolicy[Float, ActType, ObsType]]

    gae_lambda: eqx.AbstractVar[float]
    gamma: eqx.AbstractVar[float]
    num_steps: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def __check_init__(self):
        """
        Check invariants.

        Called automatically after the object is initialized by Equinox.
        """
        if (self.env.action_space != self.policy.action_space) or (
            self.env.observation_space != self.policy.observation_space
        ):
            raise ValueError(
                "The action and observation spaces of the environment and policy must match."
            )

    def step(
        self,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        state: eqx.nn.State,
        carry: StepCarry[ObsType],
        *,
        key: Key,
    ) -> tuple[eqx.nn.State, StepCarry[ObsType], RolloutBuffer[ActType, ObsType]]:
        """
        Perform a single step in the environment.
        """
        action_key, env_key, reset_key = jr.split(key, 3)

        policy_state = state.substate(self.policy)
        policy_state, action, value, log_prob = policy(
            clone_state(  # Don't consume the state since it will also be used in training
                policy_state
            ),
            carry.last_obs,
            key=action_key,
        )
        state = state.update(policy_state)

        env_state = state.substate(self.env)
        env_state, observation, reward, termination, truncation, info = self.env.step(
            env_state, action, key=env_key
        )
        state = state.update(env_state)

        def reset_env() -> (
            tuple[eqx.nn.State, ObsType, Bool[Array, ""], Bool[Array, ""], dict]
        ):
            env_state = state.substate(self.env)
            env_state, observation, info = self.env.reset(env_state, key=reset_key)
            _state = state.update(env_state)  # _ prefix to avoid shadowing `state`

            policy_state = state.substate(self.policy)
            policy_state = policy.reset(policy_state)
            _state = _state.update(policy_state)

            return _state, observation, jnp.asarray(False), jnp.asarray(False), info

        def identity() -> (
            tuple[eqx.nn.State, ObsType, Bool[Array, ""], Bool[Array, ""], dict]
        ):
            return state, observation, termination, truncation, info

        done = jnp.logical_or(termination, truncation)

        state, observation, termination, truncation, info = lax.cond(
            done,
            reset_env,
            identity,
            state,
        )

        return (
            state,
            StepCarry(carry.step_count + 1, observation, termination, truncation),
            RolloutBuffer(
                observations=carry.last_obs,
                actions=action,
                rewards=reward,
                terminations=carry.last_termination,
                truncations=carry.last_truncation,
                log_probs=log_prob,
                values=value,
                states=state.substate(self.policy),
            ),
        )

    def collect_rollout(
        self,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        state: eqx.nn.State,
        carry: StepCarry[ObsType],
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> tuple[eqx.nn.State, StepCarry[ObsType], RolloutBuffer[ActType, ObsType]]:
        """
        Collect a rollout from the environment and store it in a buffer.
        """

        def scan_step(
            carry: tuple[eqx.nn.State, StepCarry[ObsType], Key],
            _,
        ) -> tuple[
            tuple[eqx.nn.State, StepCarry[ObsType], Key],
            RolloutBuffer[ActType, ObsType],
        ]:
            state, previous, key = carry
            step_key, carry_key = jr.split(key, 2)
            state, previous, rollout = self.step(
                policy, state, previous, key=step_key, tb_writer=tb_writer
            )
            return (state, previous, carry_key), rollout

        (state, carry, _), rollout_buffer = lax.scan(
            scan_step, (state, carry, key), length=self.num_steps
        )

        next_done = jnp.logical_or(carry.last_termination, carry.last_truncation)
        _, next_value = policy.value(clone_state(state), carry.last_obs)
        rollout_buffer = rollout_buffer.compute_returns_and_advantages(
            next_value, next_done, self.gae_lambda, self.gamma
        )

        return state, carry, rollout_buffer

    @abstractmethod
    def train(
        self,
        state: eqx.nn.State,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> tuple[eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType]]:
        """
        Train the policy using the rollout buffer.
        """

    def initialize_iteration_carry(
        self,
        state: eqx.nn.State,
        *,
        key: Key,
    ) -> tuple[eqx.nn.State, IterationCarry[ActType, ObsType]]:
        env_state, next_obs, _ = self.env.reset(state.substate(self.env), key=key)
        state = state.update(env_state)

        policy_state = self.policy.reset(state.substate(self.policy))
        state = state.update(policy_state)

        return state, IterationCarry(
            StepCarry(jnp.asarray(0), next_obs, jnp.asarray(False), jnp.asarray(False)),
            self.policy,
        )

    def iteration(
        self,
        state: eqx.nn.State,
        carry: IterationCarry[ActType, ObsType],
        *,
        key: Key,
    ) -> tuple[eqx.nn.State, IterationCarry[ActType, ObsType]]:
        """
        Perform a single iteration of the algorithm.
        """
        rollout_key, train_key = jr.split(key, 2)
        state, step_carry, rollout_buffer = self.collect_rollout(
            carry.policy,
            state,
            carry.step_carry,
            key=rollout_key,
        )
        state, policy = self.train(state, carry.policy, rollout_buffer, key=train_key)
        return state, IterationCarry(step_carry, policy)

    def learn(
        self,
        state: eqx.nn.State,
        total_timesteps: int,
        *,
        key: Key,
        progress_bar: bool = False,
        tb_log_name: str | None = None,
        log_interval: int = 100,
    ):
        """
        Return a trained model.
        """

        init_key, learn_key = jr.split(key, 2)
        state, carry = self.initialize_iteration_carry(state, key=init_key)

        # TODO: Implement progress bar under JIT
        progress = create_progress_bar()
        task = progress.add_task("", total=total_timesteps)
        # TODO: Implement logging under JIT
        tb_writer = (
            SummaryWriter(log_dir=f"runs/{tb_log_name}" if tb_log_name else "runs")
            if tb_log_name
            else None
        )
        if tb_writer is not None:
            tb_writer.add_scalar = debug_with_numpy_wrapper(tb_writer.add_scalar)

        def scan_iteration(
            carry: tuple[eqx.nn.State, IterationCarry[ActType, ObsType], Key], _
        ) -> tuple[tuple[eqx.nn.State, IterationCarry[ActType, ObsType], Key], None]:
            state, iter_carry, key = carry
            iter_key, carry_key = jr.split(key, 2)

            state, iter_carry = self.iteration(state, iter_carry, key=iter_key)

            return (state, iter_carry, carry_key), None

        num_iterations = total_timesteps // self.num_steps

        (state, carry, _), _ = filter_scan(
            scan_iteration, (state, carry, learn_key), length=num_iterations
        )

        self.policy = carry.policy
