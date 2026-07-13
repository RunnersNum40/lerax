import jax.numpy as jnp
from jax import random as jr

from lerax.buffer import ReplayBuffer, RolloutBuffer
from lerax.policy import AbstractPolicyState
from lerax.space import Box, Discrete


class CounterState(AbstractPolicyState):
    value: jnp.ndarray


def test_replay_buffer_supports_stateless_policy():
    observation_space = Box(-1.0, 1.0, shape=(2,))
    action_space = Discrete(2)
    buffer = ReplayBuffer(4, observation_space, action_space, None)

    buffer = buffer.add(
        observation=jnp.array([0.25, -0.5]),
        next_observation=jnp.array([0.5, -0.25]),
        action=jnp.array(1),
        reward=1.5,
        done=False,
        timeout=True,
        state=None,
        next_state=None,
    )
    sample = buffer.sample(1, key=jr.key(0))

    assert sample.states is None
    assert sample.next_states is None
    assert jnp.array_equal(sample.observations[0], jnp.array([0.25, -0.5]))
    assert jnp.array_equal(sample.actions, jnp.array([1]))
    assert jnp.array_equal(sample.rewards, jnp.array([1.5]))
    assert jnp.array_equal(sample.timeouts, jnp.array([True]))


def test_replay_buffer_preserves_stateful_policy_state():
    observation_space = Box(-1.0, 1.0, shape=(1,))
    action_space = Discrete(2)
    buffer = ReplayBuffer(
        2, observation_space, action_space, CounterState(jnp.array(0))
    )

    buffer = buffer.add(
        observation=jnp.array([0.0]),
        next_observation=jnp.array([0.5]),
        action=jnp.array(0),
        reward=0.0,
        done=False,
        timeout=False,
        state=CounterState(jnp.array(3)),
        next_state=CounterState(jnp.array(4)),
    )
    sample = buffer.sample(1, key=jr.key(1))

    assert jnp.array_equal(sample.states.value, jnp.array([3]))
    assert jnp.array_equal(sample.next_states.value, jnp.array([4]))


def test_rollout_buffer_supports_stateless_policy_and_keeps_gae_api():
    buffer = RolloutBuffer(
        observations=jnp.array([[0.0], [1.0]]),
        actions=jnp.array([0, 1]),
        rewards=jnp.array([1.0, 1.0]),
        dones=jnp.array([False, True]),
        log_probs=jnp.array([-0.1, -0.2]),
        values=jnp.array([0.5, 0.25]),
        states=None,
        action_masks=jnp.array([[True, True], [True, False]]),
    )

    updated = buffer.compute_returns_and_advantages(
        last_value=0.0, gae_lambda=0.95, gamma=0.99
    )

    assert updated.states is None
    assert updated.action_masks is not None
    assert buffer.action_masks is not None
    assert jnp.array_equal(updated.action_masks, buffer.action_masks)
    assert jnp.all(jnp.isfinite(updated.advantages))
    assert jnp.allclose(updated.returns, updated.advantages + updated.values)
