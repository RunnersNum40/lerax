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
    Q-network for TD3.

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


class TD3StepState[PolicyType: AbstractDeterministicPolicy](AbstractStepState):
    """
    Step-level state for TD3.

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
    ) -> TD3StepState[PolicyType]:
        """Initialize the step state with an empty replay buffer."""
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        callback_state = callback.step_reset(ResetContext(locals()), key=key)
        buffer = ReplayBuffer(
            size, env.observation_space, env.action_space, policy_state
        )
        return cls(env_state, policy_state, callback_state, buffer)


class TD3State[PolicyType: AbstractDeterministicPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    Iteration-level state for TD3.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The step-level state.
        env: The environment being used.
        policy: The online policy being trained.
        opt_state: The actor optimizer state.
        callback_state: The callback state.
        target_policy: The target actor for stable Q-value estimation.
        qf1: First Q-network.
        qf2: Second Q-network.
        qf1_target: Target for first Q-network.
        qf2_target: Target for second Q-network.
        q_opt_state: The Q-network optimizer state.
    """

    iteration_count: Int[Array, ""]
    step_state: TD3StepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState
    target_policy: PolicyType
    qf1: QNetwork
    qf2: QNetwork
    qf1_target: QNetwork
    qf2_target: QNetwork
    q_opt_state: optax.OptState


class TD3[PolicyType: AbstractDeterministicPolicy](
    AbstractAlgorithm[PolicyType, TD3State[PolicyType]]
):
    """
    Twin Delayed DDPG (TD3) algorithm.

    Extends DDPG with three improvements: twin Q-networks (take the
    minimum for target computation to reduce overestimation), delayed
    policy updates (actor updates less frequently than critic), and
    target policy smoothing (clipped noise added to target actions).

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
        policy_frequency: How often to update the actor (relative to Q updates).
        policy_noise: Noise scale for target policy smoothing.
        noise_clip: Clipping range for target policy smoothing noise.
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
        policy_frequency: How often to update the actor.
        policy_noise: Noise scale for target policy smoothing.
        noise_clip: Clipping range for target policy smoothing noise.
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
    policy_frequency: int
    policy_noise: float
    noise_clip: float

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
        policy_frequency: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
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
        self.policy_frequency = policy_frequency
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.q_width_size = q_width_size
        self.q_depth = q_depth

        self.optimizer = optax.inject_hyperparams(optax.adam)(actor_lr)
        self.q_optimizer = optax.inject_hyperparams(optax.adam)(q_lr)

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: TD3StepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> TD3StepState[PolicyType]:
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

        return TD3StepState(
            next_env_state, next_policy_state, callback_state, replay_buffer
        )

    def collect_learning_starts(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: TD3StepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> TD3StepState[PolicyType]:
        """Collect random initial experience before training begins."""

        def scan_step(
            carry: TD3StepState, key: Key[Array, ""]
        ) -> tuple[TD3StepState, None]:
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
        step_state: TD3StepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> TD3StepState[PolicyType]:
        """Collect a rollout of experience into the replay buffer."""

        def scan_step(
            carry: TD3StepState, key: Key[Array, ""]
        ) -> tuple[TD3StepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return carry, None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.num_steps)
        )
        return step_state

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> TD3State[PolicyType]:
        init_key, starts_key, callback_key, qf1_key, qf2_key = jr.split(key, 5)

        if self.num_envs == 1:
            step_state = TD3StepState.initial(
                self.buffer_size, env, policy, callback, init_key
            )
            step_state = self.collect_learning_starts(
                env, policy, step_state, callback, starts_key
            )
        else:
            step_state = jax.vmap(
                TD3StepState.initial, in_axes=(None, None, None, None, 0)
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

        qf1 = QNetwork(
            observation_size,
            action_size,
            width_size=self.q_width_size,
            depth=self.q_depth,
            key=qf1_key,
        )
        qf2 = QNetwork(
            observation_size,
            action_size,
            width_size=self.q_width_size,
            depth=self.q_depth,
            key=qf2_key,
        )

        q_params = (
            eqx.filter(qf1, eqx.is_inexact_array),
            eqx.filter(qf2, eqx.is_inexact_array),
        )

        return TD3State(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
            target_policy=policy,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1,
            qf2_target=qf2,
            q_opt_state=self.q_optimizer.init(q_params),
        )

    def iteration(
        self,
        state: TD3State[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> TD3State[PolicyType]:
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

        policy, opt_state, qf1, qf2, q_opt_state, log = self.td3_train(
            state.policy,
            state.opt_state,
            step_state.buffer,
            state.qf1,
            state.qf2,
            state.qf1_target,
            state.qf2_target,
            state.target_policy,
            state.q_opt_state,
            state.iteration_count,
            key=train_key,
        )

        state = state.next(step_state, policy, opt_state)
        state = eqx.tree_at(
            lambda s: (s.qf1, s.qf2, s.q_opt_state),
            state,
            (qf1, qf2, q_opt_state),
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

        state, new_cb = callback.apply_curriculum(state, state.callback_state)
        state = state.with_callback_states(new_cb)
        return self.per_iteration(state)

    def per_iteration(self, state: TD3State[PolicyType]) -> TD3State[PolicyType]:
        """Polyak-average actor and critic target networks."""

        return eqx.tree_at(
            lambda s: (s.target_policy, s.qf1_target, s.qf2_target),
            state,
            (
                polyak_average(state.policy, state.target_policy, self.tau),
                polyak_average(state.qf1, state.qf1_target, self.tau),
                polyak_average(state.qf2, state.qf2_target, self.tau),
            ),
        )

    @staticmethod
    def q_loss(
        q_params: tuple[QNetwork, QNetwork],
        batch: ReplayBuffer,
        target: Float[Array, " batch_size"],
    ) -> Float[Array, ""]:
        """Compute combined MSE loss for twin Q-networks."""
        qf1, qf2 = q_params
        qf1_values = jax.vmap(qf1)(batch.observations, batch.actions).squeeze()
        qf2_values = jax.vmap(qf2)(batch.observations, batch.actions).squeeze()
        qf1_loss = jnp.mean(jnp.square(qf1_values - target)) / 2
        qf2_loss = jnp.mean(jnp.square(qf2_values - target)) / 2
        return qf1_loss + qf2_loss

    q_loss_grad = staticmethod(eqx.filter_value_and_grad(q_loss))

    @staticmethod
    def actor_loss(
        policy: PolicyType,
        batch: ReplayBuffer,
        qf1: QNetwork,
    ) -> Float[Array, ""]:
        """Compute actor loss: maximize Q1-value of deterministic action."""

        def single_loss(observation):
            _, action = policy(None, observation)
            return -qf1(observation, action)

        return jnp.mean(jax.vmap(single_loss)(batch.observations))

    actor_loss_grad = staticmethod(eqx.filter_value_and_grad(actor_loss))

    def td3_train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        qf1: QNetwork,
        qf2: QNetwork,
        qf1_target: QNetwork,
        qf2_target: QNetwork,
        target_policy: PolicyType,
        q_opt_state: optax.OptState,
        iteration_count: Int[Array, ""],
        *,
        key: Key[Array, ""],
    ) -> tuple[
        PolicyType,
        optax.OptState,
        QNetwork,
        QNetwork,
        optax.OptState,
        dict[str, Scalar],
    ]:
        sample_key, noise_key = jr.split(key, 2)
        batch = buffer.sample(self.batch_size, key=sample_key)

        # Target policy smoothing: add clipped noise to target actions
        action_low = policy.action_space.low  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        action_high = policy.action_space.high  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        action_scale = (action_high - action_low) / 2.0

        def compute_target(next_obs, reward, done, timeout, noise_key):
            _, next_action = target_policy(None, next_obs)
            noise = jnp.clip(
                jr.normal(noise_key, next_action.shape)
                * self.policy_noise
                * action_scale,
                -self.noise_clip * action_scale,
                self.noise_clip * action_scale,
            )
            next_action = jnp.clip(
                next_action + noise,
                action_low,
                action_high,
            )
            q1_next = qf1_target(next_obs, next_action)
            q2_next = qf2_target(next_obs, next_action)
            min_q_next = jnp.minimum(q1_next, q2_next)
            not_terminal = (~done | timeout).astype(float)
            return reward + self.gamma * min_q_next * not_terminal

        targets = jax.vmap(compute_target)(
            batch.next_observations,
            batch.rewards,
            batch.dones,
            batch.timeouts,
            jr.split(noise_key, self.batch_size),
        )

        # Update twin critics
        q_loss_val, q_grads = self.q_loss_grad((qf1, qf2), batch, targets)
        q_updates, q_opt_state = self.q_optimizer.update(
            q_grads,
            q_opt_state,
            (
                eqx.filter(qf1, eqx.is_inexact_array),
                eqx.filter(qf2, eqx.is_inexact_array),
            ),
        )
        qf1, qf2 = eqx.apply_updates((qf1, qf2), q_updates)

        # Delayed actor update
        should_update_actor = iteration_count % self.policy_frequency == 0

        def update_actor():
            _, a_grads = self.actor_loss_grad(policy, batch, qf1)
            a_updates, new_opt_state = self.optimizer.update(
                a_grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
            )
            return eqx.apply_updates(policy, a_updates), new_opt_state

        def skip_actor():
            return policy, opt_state

        policy, opt_state = filter_cond(should_update_actor, update_actor, skip_actor)

        log: dict[str, Scalar] = {"q_loss": q_loss_val}
        return policy, opt_state, qf1, qf2, q_opt_state, log
