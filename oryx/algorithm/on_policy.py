from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key, Scalar, ScalarLike
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from tensorboardX import SummaryWriter

from oryx.buffer import RolloutBuffer
from oryx.env import AbstractEnvLike
from oryx.policy import AbstractActorCriticPolicy
from oryx.utils import (
    clone_state,
    debug_with_list_wrapper,
    debug_with_numpy_wrapper,
    debug_wrapper,
    filter_scan,
)

from .base_algorithm import AbstractAlgorithm


class JITSummaryWriter:
    """
    A wrapper around `tensorboardX.SummaryWriter` with a JIT compatible interface.
    """

    summary_writer: SummaryWriter

    def __init__(self, log_dir: str | None = None):
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(
        self,
        tag: str,
        scalar_value: ScalarLike,
        global_step: Int[ArrayLike, ""] | None = None,
        walltime: Float[ArrayLike, ""] | None = None,
    ):
        """
        Add a scalar value to the summary writer.
        """
        debug_with_numpy_wrapper(self.summary_writer.add_scalar, thread=True)(
            tag, scalar_value, global_step, walltime
        )

    def add_dict(
        self,
        scalars: dict[str, Scalar],
        *,
        global_step: Int[ArrayLike, ""] | None = None,
        walltime: Float[ArrayLike, ""] | None = None,
    ) -> None:
        """
        Log a dictionary of **scalar** values.
        """

        for tag, value in scalars.items():
            self.add_scalar(tag, value, global_step=global_step, walltime=walltime)


class JITProgressBar:
    progress_bar: Progress
    task: TaskID

    def __init__(self, name: str, total: int | None):
        self.progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        self.task = self.progress_bar.add_task(name, total=total)

    def update(
        self,
        total: Float[ArrayLike, ""] | None = None,
        completed: Float[ArrayLike, ""] | None = None,
        advance: Float[ArrayLike, ""] | None = None,
        description: str | None = None,
        visible: Bool[ArrayLike, ""] | None = None,
        refresh: Bool[ArrayLike, ""] = False,
    ) -> None:
        debug_with_list_wrapper(self.progress_bar.update)(
            self.task,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
        )


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
    ) -> tuple[eqx.nn.State, StepCarry[ObsType], RolloutBuffer[ActType, ObsType], dict]:
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

        def reset_env(
            state: eqx.nn.State,
        ) -> tuple[eqx.nn.State, ObsType, Bool[Array, ""], Bool[Array, ""], dict]:
            """
            Reset the environment and policy states.

            This is called when the episode is done.
            """
            env_state = state.substate(self.env)
            env_state, reset_observation, reset_info = self.env.reset(
                env_state, key=reset_key
            )
            state = state.update(env_state)

            policy_state = state.substate(self.policy)
            policy_state = policy.reset(policy_state)
            state = state.update(policy_state)

            return (
                state,
                reset_observation,
                jnp.asarray(False),
                jnp.asarray(False),
                reset_info,
            )

        def identity(
            state: eqx.nn.State,
        ) -> tuple[eqx.nn.State, ObsType, Bool[Array, ""], Bool[Array, ""], dict]:
            """
            Return the current state.

            Matches the signature of `reset_env` for use in `lax.cond`.
            """
            return state, observation, termination, truncation, info

        done = jnp.logical_or(termination, truncation)

        state, observation, termination, truncation, info = lax.cond(
            done, reset_env, identity, state
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
            info,
        )

    def collect_rollout(
        self,
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
        state: eqx.nn.State,
        carry: StepCarry[ObsType],
        *,
        key: Key,
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
            state, previous, rollout, info = self.step(
                policy, state, previous, key=step_key
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
    ) -> tuple[
        eqx.nn.State,
        AbstractActorCriticPolicy[Float, ActType, ObsType],
        dict[str, Scalar],
    ]:
        """
        Train the policy using the rollout buffer.
        """

    @abstractmethod
    def learning_rate(self, state: eqx.nn.State) -> Float[Array, ""]:
        """
        Return the current learning rate.
        This is used for logging purposes.
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
        progress_bar: JITProgressBar | None,
        tb_writer: JITSummaryWriter | None,
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
        state, policy, log = self.train(
            state, carry.policy, rollout_buffer, key=train_key
        )

        if progress_bar is not None:
            progress_bar.update(advance=self.num_steps)
        if tb_writer is not None:
            log["loss/learning_rate"] = self.learning_rate(state)

            tb_writer.add_dict(log, global_step=carry.step_carry.step_count)

        debug_wrapper(print)(carry.step_carry.step_count)

        return state, IterationCarry(step_carry, policy)

    def learn(
        self,
        state: eqx.nn.State,
        total_timesteps: int,
        *,
        key: Key,
        show_progress_bar: bool = False,
        tb_log_name: str | None = None,
    ) -> tuple[eqx.nn.State, AbstractActorCriticPolicy[Float, ActType, ObsType]]:
        """
        Return a trained model.
        """

        init_key, learn_key = jr.split(key, 2)
        state, carry = self.initialize_iteration_carry(state, key=init_key)

        progress_bar = (
            JITProgressBar("Training", total=total_timesteps)
            if show_progress_bar
            else None
        )
        tb_writer = JITSummaryWriter(tb_log_name) if tb_log_name is not None else None

        def scan_iteration(
            carry: tuple[eqx.nn.State, IterationCarry[ActType, ObsType], Key], _
        ) -> tuple[tuple[eqx.nn.State, IterationCarry[ActType, ObsType], Key], None]:
            state, iter_carry, key = carry
            iter_key, carry_key = jr.split(key, 2)

            state, iter_carry = self.iteration(
                state,
                iter_carry,
                key=iter_key,
                progress_bar=progress_bar,
                tb_writer=tb_writer,
            )

            return (state, iter_carry, carry_key), None

        num_iterations = total_timesteps // self.num_steps

        (state, carry, _), _ = filter_scan(
            scan_iteration, (state, carry, learn_key), length=num_iterations
        )

        return state, carry.policy
