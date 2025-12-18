import jax
import jax.numpy as jnp
import pytest

from lerax.distribution import MultiCategorical


def test_init_sequence_logits_sets_action_dims_and_matches_flat():
    logits_seq = [
        jnp.array([0.1, 0.2, 0.3]),
        jnp.array([-1.0, 2.0]),
    ]
    mc_seq = MultiCategorical(logits=logits_seq)

    assert mc_seq.action_dims == (3, 2)
    assert len(mc_seq.distribution) == 2
    assert mc_seq.logits.shape == (5,)

    logits_flat = jnp.concatenate(logits_seq, axis=-1)
    mc_flat = MultiCategorical(logits=logits_flat, action_dims=(3, 2))

    assert mc_flat.action_dims == (3, 2)
    assert jnp.allclose(mc_flat.logits, mc_seq.logits)
    assert jnp.allclose(mc_flat.probs, mc_seq.probs)


def test_init_with_probs_sequence_and_flat():
    probs_seq = [
        jnp.array([0.2, 0.3, 0.5]),
        jnp.array([0.7, 0.3]),
    ]
    mc_seq = MultiCategorical(probs=probs_seq)
    assert mc_seq.action_dims == (3, 2)

    probs_flat = jnp.concatenate(probs_seq, axis=-1)
    mc_flat = MultiCategorical(probs=probs_flat, action_dims=(3, 2))

    assert jnp.allclose(mc_flat.probs, mc_seq.probs)


def test_log_prob_is_sum_of_components_with_batch():
    batch = 5
    logits_a = jnp.arange(batch * 3, dtype=jnp.float32).reshape(batch, 3) * 0.1
    logits_b = jnp.arange(batch * 4, dtype=jnp.float32).reshape(batch, 4) * -0.2
    mc = MultiCategorical(logits=[logits_a, logits_b])

    value = jnp.stack(
        [
            jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32),
            jnp.array([3, 2, 1, 0, 3], dtype=jnp.int32),
        ],
        axis=-1,
    )
    got = mc.log_prob(value)

    expected = mc.distribution[0].log_prob(value[..., 0]) + mc.distribution[1].log_prob(
        value[..., 1]
    )

    assert got.shape == (batch,)
    assert jnp.allclose(got, expected)

    got_jit = jax.jit(lambda v: mc.log_prob(v))(value)
    assert jnp.allclose(got_jit, expected)


def test_sample_shape_and_value_ranges_with_batch():
    batch = 7
    logits_a = jnp.zeros((batch, 3), dtype=jnp.float32)
    logits_b = jnp.zeros((batch, 5), dtype=jnp.float32)
    mc = MultiCategorical(logits=[logits_a, logits_b])

    key = jax.random.key(0)
    sample = mc.sample(key)

    assert sample.shape == (batch, 2)
    assert jnp.all((sample[..., 0] >= 0) & (sample[..., 0] < 3))
    assert jnp.all((sample[..., 1] >= 0) & (sample[..., 1] < 5))


def test_sample_and_log_prob_consistent():
    logits_seq = [
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([3.0, 4.0]),
    ]
    mc = MultiCategorical(logits=logits_seq)

    key = jax.random.key(123)
    sample, lp = mc.sample_and_log_prob(key)

    assert sample.shape == (2,)
    assert lp.shape == ()
    assert jnp.allclose(lp, mc.log_prob(sample))


def test_mask_applies_independently_sequence_and_flat():
    logits_seq = [
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([-1.0, 0.5]),
    ]
    mc = MultiCategorical(logits=logits_seq)

    mask_seq = [
        jnp.array([True, False, True]),
        jnp.array([False, True]),
    ]
    masked_seq = mc.mask(mask_seq)

    assert jnp.isneginf(masked_seq.distribution[0].logits[1])
    assert jnp.isneginf(masked_seq.distribution[1].logits[0])

    bad_action = jnp.array([1, 0], dtype=jnp.int32)
    assert jnp.isneginf(masked_seq.log_prob(bad_action))

    mask_flat = jnp.concatenate(mask_seq, axis=-1)
    masked_flat = mc.mask(mask_flat)

    assert jnp.allclose(masked_flat.logits, masked_seq.logits)


def test_bad_inputs():
    with pytest.raises(ValueError):
        MultiCategorical()

    with pytest.raises(ValueError):
        MultiCategorical(logits=jnp.array([0.0, 1.0]), probs=jnp.array([0.5, 0.5]))

    with pytest.raises(ValueError):
        MultiCategorical(logits=jnp.array([0.0, 1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        MultiCategorical(logits=jnp.array([0.0, 1.0, 2.0]), action_dims=(2, 2))

    with pytest.raises(ValueError):
        MultiCategorical(logits=[jnp.zeros((2, 3)), jnp.zeros((3, 4))])

    with pytest.raises(ValueError):
        MultiCategorical(logits=[jnp.zeros((3,)), jnp.zeros((2,))], action_dims=(3, 3))

    mc = MultiCategorical(logits=[jnp.zeros((3,)), jnp.zeros((2,))])
    with pytest.raises(ValueError):
        mc.log_prob(jnp.array([0, 1, 0], dtype=jnp.int32))
