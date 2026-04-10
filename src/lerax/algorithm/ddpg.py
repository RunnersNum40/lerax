from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    IterationContext,
    ResetContext,
    StepContext,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicyState
from lerax.policy.deterministic import AbstractDeterministicPolicy
from lerax.space import Box
from lerax.utils import filter_cond, filter_scan, polyak_average

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class QNetwork(eqx.Module):
    """
    Q-network for DDPG.

    Maps concatenated (observation, action) pairs to scalar Q-values.

    Attributes:
        mlp: The MLP that processes the concatenated input.

    Args:
        observation_size: Dimensionality of flat observations.
        action_size: Dimensionality of flat actions.
        width_size: Width of the hidden layers.
        depth: Number of hidden layers.
        key: JAX PRNG key for parameter initialization.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        width_size: int = 256,
        depth: int = 2,
        key: Key[Array, ""],
    ):
        self.mlp = eqx.nn.MLP(
            in_size=observation_size + action_size,
            out_size="scalar",
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(
        self,
        observation: Float[Array, " obs_dim"],
        action: Float[Array, " act_dim"],
    ) -> Float[Array, ""]:
        """Compute Q-value for an observation-action pair."""
        inputs = jnp.concatenate([observation.ravel(), action.ravel()])
        return self.mlp(inputs)


class DDPGStepState[PolicyType: AbstractDeterministicPolicy](AbstractStepState):
    """
    Step-level state for DDPG.

    Attributes:
        env_state: The state of the environment.
        policy_state: The state of the policy.
        callback_state: The state of the callback for this step.
        buffer: The replay buffer storing experience.
    """

    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    callback_state: AbstractCallbackStepState
    buffer: ReplayBuffer

    @classmethod
    def initial(
        cls,
        size: int,
        env: AbstractEnvLike,
        policy: PolicyType,
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> DDPGStepState[PolicyType]:
        """Initialize the step state with an empty replay buffer."""
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        callback_state = callback.step_reset(ResetContext(locals()), key=key)
        buffer = ReplayBuffer(
            size, env.observation_space, env.action_space, policy_state
        )
        return cls(env_state, policy_state, callback_state, buffer)


class DDPGState[PolicyType: AbstractDeterministicPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    Iteration-level state for DDPG.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The step-level state.
        env: The environment being used.
        policy: The online policy being trained.
        opt_state: The actor optimizer state.
        callback_state: The callback state.
        target_policy: The target actor for stable Q-value estimation.
        qf: The Q-network.
        qf_target: The target Q-network.
        q_opt_state: The Q-network optimizer state.
    """

    iteration_count: Int[Array, ""]
    step_state: DDPGStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState
    target_policy: PolicyType
    qf: QNetwork
    qf_target: QNetwork
    q_opt_state: optax.OptState


class DDPG[PolicyType: AbstractDeterministicPolicy](
    AbstractAlgorithm[PolicyType, DDPGState[PolicyType]]
):
    """
    Deep Deterministic Policy Gradient (DDPG) algorithm.

    An off-policy algorithm for continuous action spaces that learns a
    deterministic policy and a Q-function simultaneously. Uses Polyak-averaged
    target networks for both actor and critic.

    Attributes:
        optimizer: The actor optimizer.
        q_optimizer: The Q-network optimizer.
        buffer_size: The size of the replay buffer.
        gamma: Discount factor for future rewards.
        learning_starts: Number of initial steps to collect before training.
        num_envs: Number of parallel environments.
        num_steps: Number of steps per iteration.
        batch_size: Batch size for training.
        tau: Soft update coefficient for target networks.
        q_width_size: Width of Q-network hidden layers.
        q_depth: Depth of Q-network hidden layers.

    Args:
        buffer_size: The size of the replay buffer.
        gamma: Discount factor for future rewards.
        learning_starts: Number of initial steps to collect before training.
        num_envs: Number of parallel environments.
        num_steps: Number of steps per iteration.
        batch_size: Batch size for training.
        tau: Soft update coefficient for target networks.
        actor_lr: Learning rate for the actor.
        q_lr: Learning rate for Q-networks.
        q_width_size: Width of Q-network hidden layers.
        q_depth: Depth of Q-network hidden layers.
    """

    optimizer: optax.GradientTransformation
    q_optimizer: optax.GradientTransformation = eqx.field(static=True)

    buffer_size: int
    gamma: float
    learning_starts: int

    num_envs: int
    num_steps: int
    batch_size: int

    tau: float

    q_width_size: int
    q_depth: int

    def __init__(
        self,
        *,
        buffer_size: int = 1_000_000,
        gamma: float = 0.99,
        learning_starts: int = 25_000,
        num_envs: int = 1,
        num_steps: int = 1,
        batch_size: int = 256,
        tau: float = 0.005,
        actor_lr: optax.ScalarOrSchedule = 3e-4,
        q_lr: optax.ScalarOrSchedule = 3e-4,
        q_width_size: int = 256,
        q_depth: int = 2,
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.learning_starts = learning_starts

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.tau = tau
        self.q_width_size = q_width_size
        self.q_depth = q_depth

        self.optimizer = optax.inject_hyperparams(optax.adam)(actor_lr)
        self.q_optimizer = optax.inject_hyperparams(optax.adam)(q_lr)

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    # ── Step & rollout collection ──────────────────────────────────────

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: DDPGStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> DDPGStepState[PolicyType]:
        """Perform a single environment step and store in replay buffer."""
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            next_observation_key,
            env_reset_key,
            policy_reset_key,
            callback_key,
        ) = jr.split(key, 9)

        observation = env.observation(state.env_state, key=observation_key)
        policy_state, action = policy(state.policy_state, observation, key=action_key)

        if isinstance(env.action_space, Box):
            clipped_action = jnp.clip(
                action, env.action_space.low, env.action_space.high
            )
        else:
            clipped_action = action

        next_env_state = env.transition(
            state.env_state, clipped_action, key=transition_key
        )

        reward = env.reward(state.env_state, action, next_env_state, key=reward_key)
        termination = env.terminal(next_env_state, key=terminal_key)
        truncation = env.truncate(next_env_state)
        done = termination | truncation
        timeout = truncation & ~termination
        next_observation = env.observation(next_env_state, key=next_observation_key)

        next_env_state = filter_cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )
        next_policy_state = filter_cond(
            done, lambda: policy.reset(key=policy_reset_key), lambda: policy_state
        )

        replay_buffer = state.buffer.add(
            observation,
            next_observation,
            action,
            reward,
            done,
            timeout,
            state.policy_state,
            policy_state,
        )

        callback_state = callback.on_step(
            StepContext(state.callback_state, env, policy, done, reward, locals()),
            key=callback_key,
        )

        return DDPGStepState(
            next_env_state, next_policy_state, callback_state, replay_buffer
        )

    def collect_learning_starts(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: DDPGStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> DDPGStepState[PolicyType]:
        """Collect random initial experience before training begins."""

        def scan_step(
            carry: DDPGStepState, key: Key[Array, ""]
        ) -> tuple[DDPGStepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return carry, None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.learning_starts)
        )
        return step_state

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: DDPGStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> DDPGStepState[PolicyType]:
        """Collect a rollout of experience into the replay buffer."""

        def scan_step(
            carry: DDPGStepState, key: Key[Array, ""]
        ) -> tuple[DDPGStepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return carry, None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.num_steps)
        )
        return step_state

    # ── Reset & iteration ──────────────────────────────────────────────

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> DDPGState[PolicyType]:
        init_key, starts_key, callback_key, qf_key = jr.split(key, 4)

        if self.num_envs == 1:
            step_state = DDPGStepState.initial(
                self.buffer_size, env, policy, callback, init_key
            )
            step_state = self.collect_learning_starts(
                env, policy, step_state, callback, starts_key
            )
        else:
            step_state = jax.vmap(
                DDPGStepState.initial, in_axes=(None, None, None, None, 0)
            )(
                self.buffer_size // self.num_envs,
                env,
                policy,
                callback,
                jr.split(init_key, self.num_envs),
            )
            step_state = jax.vmap(
                self.collect_learning_starts, in_axes=(None, None, 0, None, 0)
            )(env, policy, step_state, callback, jr.split(starts_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        observation_size = env.observation_space.flat_size
        action_size = max(env.action_space.flat_size, 1)

        qf = QNetwork(
            observation_size,
            action_size,
            width_size=self.q_width_size,
            depth=self.q_depth,
            key=qf_key,
        )

        return DDPGState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
            target_policy=policy,
            qf=qf,
            qf_target=qf,
            q_opt_state=self.q_optimizer.init(eqx.filter(qf, eqx.is_inexact_array)),
        )

    def iteration(
        self,
        state: DDPGState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> DDPGState[PolicyType]:
        rollout_key, train_key, callback_key = jr.split(key, 3)

        if self.num_envs == 1:
            step_state = self.collect_rollout(
                state.env, state.policy, state.step_state, callback, rollout_key
            )
        else:
            step_state = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, eqx.if_array(0), None, 0)
            )(
                state.env,
                state.policy,
                state.step_state,
                callback,
                jr.split(rollout_key, self.num_envs),
            )

        policy, opt_state, qf, q_opt_state, log = self.ddpg_train(
            state.policy,
            state.opt_state,
            step_state.buffer,
            state.qf,
            state.qf_target,
            state.target_policy,
            state.q_opt_state,
            key=train_key,
        )

        state = state.next(step_state, policy, opt_state)
        state = eqx.tree_at(
            lambda s: (s.qf, s.q_opt_state),
            state,
            (qf, q_opt_state),
        )

        state = state.with_callback_states(
            callback.on_iteration(
                IterationContext(
                    state.callback_state,
                    state.step_state.callback_state,
                    state.env,
                    state.policy,
                    state.iteration_count,
                    state.opt_state,
                    log,
                    self,
                    locals(),
                ),
                key=callback_key,
            )
        )

        return self.per_iteration(state)

    def per_iteration(self, state: DDPGState[PolicyType]) -> DDPGState[PolicyType]:
        """Polyak-average both actor and critic target networks."""

        return eqx.tree_at(
            lambda s: (s.target_policy, s.qf_target),
            state,
            (
                polyak_average(state.policy, state.target_policy, self.tau),
                polyak_average(state.qf, state.qf_target, self.tau),
            ),
        )

    # ── Training ───────────────────────────────────────────────────────

    @staticmethod
    def q_loss(
        qf: QNetwork,
        batch: ReplayBuffer,
        target: Float[Array, " batch_size"],
    ) -> Float[Array, ""]:
        """Compute MSE loss for Q-network."""
        q_values = jax.vmap(qf)(batch.observations, batch.actions).squeeze()
        return jnp.mean(jnp.square(q_values - target)) / 2

    q_loss_grad = staticmethod(eqx.filter_value_and_grad(q_loss))

    @staticmethod
    def actor_loss(
        policy: PolicyType,
        batch: ReplayBuffer,
        qf: QNetwork,
    ) -> Float[Array, ""]:
        """Compute actor loss: maximize Q-value of deterministic action."""

        def single_loss(observation):
            _, action = policy(None, observation)
            return -qf(observation, action)

        return jnp.mean(jax.vmap(single_loss)(batch.observations))

    actor_loss_grad = staticmethod(eqx.filter_value_and_grad(actor_loss))

    def ddpg_train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        qf: QNetwork,
        qf_target: QNetwork,
        target_policy: PolicyType,
        q_opt_state: optax.OptState,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, QNetwork, optax.OptState, dict[str, Scalar]]:
        batch = buffer.sample(self.batch_size, key=key)

        # Compute Q-targets using target actor and target critic
        def compute_target(next_obs, reward, done, timeout):
            _, next_action = target_policy(None, next_obs)
            q_next = qf_target(next_obs, next_action)
            not_terminal = (~done | timeout).astype(float)
            return reward + self.gamma * q_next * not_terminal

        targets = jax.vmap(compute_target)(
            batch.next_observations, batch.rewards, batch.dones, batch.timeouts
        )

        # Update critic
        q_loss_val, q_grads = self.q_loss_grad(qf, batch, targets)
        q_updates, q_opt_state = self.q_optimizer.update(
            q_grads, q_opt_state, eqx.filter(qf, eqx.is_inexact_array)
        )
        qf = eqx.apply_updates(qf, q_updates)

        # Update actor
        _, a_grads = self.actor_loss_grad(policy, batch, qf)
        a_updates, opt_state = self.optimizer.update(
            a_grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, a_updates)

        log: dict[str, Scalar] = {"q_loss": q_loss_val}
        return policy, opt_state, qf, q_opt_state, log
