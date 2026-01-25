import pytest
from jax import numpy as jnp
from jax import random as jr

from lerax.distribution import MultiCategorical


class TestMultiCategorical:
    """Tests for the MultiCategorical distribution."""

    def test_init_with_sequence_logits(self):
        """Test initialization with sequence of logits."""
        logits = [jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0, 3.0])]
        dist = MultiCategorical(logits=logits)

        assert dist.action_dims == (2, 3)
        assert len(dist.distribution) == 2

    def test_init_with_flat_logits(self):
        """Test initialization with flat array and action_dims."""
        logits = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        dist = MultiCategorical(logits=logits, action_dims=[2, 3])

        assert dist.action_dims == (2, 3)
        assert len(dist.distribution) == 2

    def test_init_with_sequence_probs(self):
        """Test initialization with sequence of probabilities."""
        probs = [jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])]
        dist = MultiCategorical(probs=probs)

        assert dist.action_dims == (2, 3)

    def test_init_with_flat_probs(self):
        """Test initialization with flat probabilities and action_dims."""
        probs = jnp.array([0.4, 0.6, 0.2, 0.3, 0.5])
        dist = MultiCategorical(probs=probs, action_dims=[2, 3])

        assert dist.action_dims == (2, 3)

    def test_init_requires_exactly_one_of_logits_or_probs(self):
        """Test that exactly one of logits or probs must be provided."""
        with pytest.raises(ValueError):
            MultiCategorical()

        with pytest.raises(ValueError):
            MultiCategorical(
                logits=jnp.array([0.0, 1.0]),
                probs=jnp.array([0.5, 0.5]),
                action_dims=[2],
            )

    def test_init_validates_action_dims(self):
        """Test action_dims validation."""
        with pytest.raises(ValueError):
            MultiCategorical(logits=jnp.array([0.0]), action_dims=[])

        with pytest.raises(ValueError):
            MultiCategorical(logits=jnp.array([0.0, 1.0]), action_dims=[0, 2])

        with pytest.raises(ValueError):
            MultiCategorical(logits=jnp.array([0.0, 1.0]), action_dims=[-1, 3])

    def test_init_requires_action_dims_for_flat_array(self):
        """Test that action_dims is required for flat array input."""
        with pytest.raises(ValueError):
            MultiCategorical(logits=jnp.array([0.0, 1.0, 2.0]))

    def test_init_validates_flat_array_shape(self):
        """Test that flat array shape matches action_dims."""
        with pytest.raises(ValueError):
            MultiCategorical(logits=jnp.array([0.0, 1.0, 2.0]), action_dims=[2, 3])

    def test_init_validates_sequence_shape_consistency(self):
        """Test that sequence elements have consistent batch shapes."""
        with pytest.raises(ValueError):
            MultiCategorical(
                logits=[jnp.array([[0.0, 1.0]]), jnp.array([1.0, 2.0, 3.0])]
            )

    def test_logits_property(self):
        """Test concatenated logits property shape."""
        logits = [jnp.array([0.0, 1.0]), jnp.array([2.0, 3.0, 4.0])]
        dist = MultiCategorical(logits=logits)

        assert dist.logits.shape == (5,)
        assert jnp.allclose(dist.logits[1] - dist.logits[0], 1.0)
        assert jnp.allclose(dist.logits[3] - dist.logits[2], 1.0)
        assert jnp.allclose(dist.logits[4] - dist.logits[3], 1.0)

    def test_probs_property(self):
        """Test concatenated probs property."""
        probs = [jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])]
        dist = MultiCategorical(probs=probs)

        expected = jnp.array([0.4, 0.6, 0.2, 0.3, 0.5])
        assert jnp.allclose(dist.probs, expected)

    def test_sample_shape(self):
        """Test that samples have the correct shape."""
        logits = [jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0, 3.0])]
        dist = MultiCategorical(logits=logits)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (2,)
        assert jnp.issubdtype(sample.dtype, jnp.integer)

    def test_sample_in_valid_range(self):
        """Test that samples are in valid category ranges."""
        logits = [jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0, 3.0])]
        dist = MultiCategorical(logits=logits)

        keys = jr.split(jr.key(0), 100)
        for k in keys:
            sample = dist.sample(k)
            assert 0 <= sample[0] < 2
            assert 0 <= sample[1] < 3

    def test_sample_and_log_prob(self):
        """Test sample_and_log_prob returns consistent values."""
        logits = [jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0, 3.0])]
        dist = MultiCategorical(logits=logits)

        key = jr.key(0)
        sample, log_prob = dist.sample_and_log_prob(key)

        expected_log_prob = dist.log_prob(sample)
        assert jnp.allclose(log_prob, expected_log_prob)

    def test_log_prob(self):
        """Test log probability is sum of individual log probs."""
        probs = [jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])]
        dist = MultiCategorical(probs=probs)

        value = jnp.array([0, 2])
        log_prob = dist.log_prob(value)

        expected = jnp.log(0.4) + jnp.log(0.5)
        assert jnp.allclose(log_prob, expected)

    def test_log_prob_validates_shape(self):
        """Test that log_prob validates input shape."""
        logits = [jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0, 3.0])]
        dist = MultiCategorical(logits=logits)

        with pytest.raises(ValueError):
            dist.log_prob(jnp.array([0, 1, 2]))

    def test_prob(self):
        """Test probability computation."""
        probs = [jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])]
        dist = MultiCategorical(probs=probs)

        value = jnp.array([0, 2])
        prob = dist.prob(value)

        expected = 0.4 * 0.5
        assert jnp.allclose(prob, expected)

    def test_entropy(self):
        """Test entropy is sum of individual entropies."""
        probs = [jnp.array([0.5, 0.5]), jnp.array([1 / 3, 1 / 3, 1 / 3])]
        dist = MultiCategorical(probs=probs)

        entropy = dist.entropy()
        expected = jnp.log(2.0) + jnp.log(3.0)
        assert jnp.allclose(entropy, expected)

    def test_mean_not_implemented(self):
        """Test that mean raises NotImplementedError for MultiCategorical."""
        probs = [jnp.array([0.2, 0.8]), jnp.array([0.1, 0.2, 0.7])]
        dist = MultiCategorical(probs=probs)

        with pytest.raises(NotImplementedError):
            dist.mean()

    def test_mode(self):
        """Test mode computation."""
        probs = [jnp.array([0.2, 0.8]), jnp.array([0.1, 0.7, 0.2])]
        dist = MultiCategorical(probs=probs)

        mode = dist.mode()
        expected = jnp.array([1, 1])
        assert jnp.array_equal(mode, expected)

    def test_mask_with_sequence(self):
        """Test masking with sequence of masks."""
        logits = [jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0, 3.0])]
        dist = MultiCategorical(logits=logits)

        mask = [jnp.array([True, False]), jnp.array([True, True, False])]
        masked_dist = dist.mask(mask)

        assert jnp.isfinite(masked_dist.distribution[0].logits[0])
        assert jnp.isneginf(masked_dist.distribution[0].logits[1])
        assert jnp.isfinite(masked_dist.distribution[1].logits[0])
        assert jnp.isfinite(masked_dist.distribution[1].logits[1])
        assert jnp.isneginf(masked_dist.distribution[1].logits[2])

    def test_mask_with_flat_array(self):
        """Test masking with flat mask array."""
        logits = jnp.array([1.0, 2.0, 1.0, 2.0, 3.0])
        dist = MultiCategorical(logits=logits, action_dims=[2, 3])

        mask = jnp.array([True, False, True, True, False])
        masked_dist = dist.mask(mask)

        assert jnp.isfinite(masked_dist.distribution[0].logits[0])
        assert jnp.isneginf(masked_dist.distribution[0].logits[1])

    def test_mask_affects_mode(self):
        """Test that masking affects the mode."""
        logits = [jnp.array([1.0, 10.0]), jnp.array([1.0, 10.0, 2.0])]
        dist = MultiCategorical(logits=logits)

        mode = dist.mode()
        assert mode[0] == 1
        assert mode[1] == 1

        mask = [jnp.array([True, False]), jnp.array([True, False, True])]
        masked_dist = dist.mask(mask)

        masked_mode = masked_dist.mode()
        assert masked_mode[0] == 0
        assert masked_mode[1] == 2

    def test_batched_operations(self):
        """Test that operations work with batched inputs."""
        logits = [
            jnp.array([[0.0, 1.0], [1.0, 0.0]]),
            jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        ]
        dist = MultiCategorical(logits=logits)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (2, 2)

    def test_batched_log_prob(self):
        """Test log_prob with batched values."""
        logits = [
            jnp.array([[0.0, 1.0], [1.0, 0.0]]),
            jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        ]
        dist = MultiCategorical(logits=logits)

        values = jnp.array([[0, 2], [1, 0]])
        log_prob = dist.log_prob(values)

        assert log_prob.shape == (2,)
