from __future__ import annotations

import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.buffer import RolloutBuffer


def _make_dummy_buffer(num_steps: int = 8):
    """Return a minimal RolloutBuffer instance with deterministic contents."""
    key = jr.key(0)
    observations = jnp.arange(num_steps) * 10.0
    actions = jnp.arange(num_steps) * -1.0
    rewards = jnp.arange(num_steps)
    terminations = jnp.full(num_steps, False)
    truncations = jnp.zeros_like(terminations)
    log_probs = jnp.zeros_like(rewards) - 0.5
    values = jnp.linspace(0.0, 1.0, num_steps)
    states = jnp.zeros(num_steps)

    return (
        RolloutBuffer(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            log_probs=log_probs,
            values=values,
            states=states,
        ),
        key,
    )


def test_buffer_shape_property():
    buf, _ = _make_dummy_buffer(11)
    assert buf.shape == (11,), "shape property should match `rewards.shape`"


def test_returns_and_advantages_are_populated_and_consistent():
    rewards = jnp.asarray([1.0, 2.0])
    values = jnp.asarray([0.5, 0.6])
    terminations = jnp.asarray([False, False])
    truncations = jnp.asarray([False, False])

    buf = RolloutBuffer(
        observations=jnp.zeros_like(rewards),
        actions=jnp.zeros_like(rewards),
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        log_probs=jnp.zeros_like(rewards),
        values=values,
        states=jnp.zeros_like(rewards),
    )

    last_value = jnp.asarray(0.7)
    done = jnp.asarray(False)

    gamma, lam = 0.99, 0.95
    buf = buf.compute_returns_and_advantages(
        last_value, done, gae_lambda=lam, gamma=gamma
    )

    assert not jnp.isnan(buf.advantages).any()
    assert not jnp.isnan(buf.returns).any()
    assert buf.advantages.shape == rewards.shape
    assert buf.returns.shape == rewards.shape

    assert jnp.allclose(buf.returns, buf.values + buf.advantages)

    expected_adv = jnp.asarray(
        [
            1.094 + 0.99 * 0.95 * 2.093,
            2.093,
        ]
    )
    expected_ret = expected_adv + values
    assert jnp.allclose(buf.advantages, expected_adv)
    assert jnp.allclose(buf.returns, expected_ret)


@pytest.mark.parametrize("shuffle", [False, True])
def test_batches_shape_and_permutation(shuffle: bool):
    n_steps, batch_sz = 12, 4
    buf, key = _make_dummy_buffer(n_steps)

    batched = buf.batches(batch_sz, key=key if shuffle else None)

    assert batched.rewards.shape == (n_steps // batch_sz, batch_sz)

    flat_after = batched.rewards.ravel()
    assert jnp.array_equal(jnp.sort(flat_after), buf.rewards)

    if shuffle:
        assert not jnp.array_equal(flat_after, buf.rewards)
    else:
        assert jnp.array_equal(flat_after, buf.rewards)


def test_batches_handles_non_divisible_batch_size():
    buf, key = _make_dummy_buffer(10)
    batch_sz = 3
    batched = buf.batches(batch_sz, key=key)
    expected_batches = 10 // batch_sz
    assert batched.rewards.shape[0] == expected_batches
