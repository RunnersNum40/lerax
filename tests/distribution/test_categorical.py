import pytest
from jax import numpy as jnp
from jax import random as jr

from lerax.distribution import Categorical


class TestCategorical:
    """Tests for the Categorical distribution."""

    def test_init_with_logits(self):
        """Test initialization with logits."""
        logits = jnp.array([0.0, 1.0, 2.0])
        dist = Categorical(logits=logits)

        assert dist.logits.shape == (3,)

    def test_init_with_probs(self):
        """Test initialization with probabilities."""
        probs = jnp.array([0.2, 0.3, 0.5])
        dist = Categorical(probs=probs)

        assert dist.probs.shape == (3,)
        assert jnp.allclose(dist.probs, probs)

    def test_sample_shape(self):
        """Test that samples have the correct shape."""
        logits = jnp.array([0.0, 1.0, 2.0])
        dist = Categorical(logits=logits)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == ()
        assert jnp.issubdtype(sample.dtype, jnp.integer)

    def test_sample_in_valid_range(self):
        """Test that samples are in valid category range."""
        logits = jnp.array([0.0, 1.0, 2.0, 3.0])
        dist = Categorical(logits=logits)

        key = jr.key(0)
        keys = jr.split(key, 100)

        for k in keys:
            sample = dist.sample(k)
            assert 0 <= sample < 4

    def test_sample_and_log_prob(self):
        """Test sample_and_log_prob returns consistent values."""
        logits = jnp.array([0.0, 1.0, 2.0])
        dist = Categorical(logits=logits)

        key = jr.key(0)
        sample, log_prob = dist.sample_and_log_prob(key)

        expected_log_prob = dist.log_prob(sample)
        assert jnp.allclose(log_prob, expected_log_prob)

    def test_log_prob(self):
        """Test log probability computation."""
        probs = jnp.array([0.25, 0.25, 0.5])
        dist = Categorical(probs=probs)

        log_prob_0 = dist.log_prob(jnp.array(0))
        log_prob_2 = dist.log_prob(jnp.array(2))

        assert jnp.allclose(log_prob_0, jnp.log(0.25))
        assert jnp.allclose(log_prob_2, jnp.log(0.5))

    def test_prob(self):
        """Test probability computation."""
        probs = jnp.array([0.25, 0.25, 0.5])
        dist = Categorical(probs=probs)

        prob_0 = dist.prob(jnp.array(0))
        prob_2 = dist.prob(jnp.array(2))

        assert jnp.allclose(prob_0, 0.25)
        assert jnp.allclose(prob_2, 0.5)

    def test_entropy(self):
        """Test entropy computation."""
        probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        dist = Categorical(probs=probs)

        entropy = dist.entropy()
        assert jnp.allclose(entropy, jnp.log(4.0))

    def test_entropy_deterministic(self):
        """Test that deterministic distribution has zero entropy."""
        probs = jnp.array([1.0, 0.0, 0.0])
        dist = Categorical(probs=probs)

        entropy = dist.entropy()
        assert jnp.allclose(entropy, 0.0, atol=1e-6)

    def test_mean_not_implemented(self):
        """Test that mean raises NotImplementedError for Categorical."""
        probs = jnp.array([0.1, 0.2, 0.7])
        dist = Categorical(probs=probs)

        with pytest.raises(NotImplementedError):
            dist.mean()

    def test_mode(self):
        """Test mode computation."""
        probs = jnp.array([0.1, 0.2, 0.7])
        dist = Categorical(probs=probs)

        mode = dist.mode()
        assert mode == 2

    def test_mask(self):
        """Test masking functionality."""
        logits = jnp.array([1.0, 2.0, 3.0])
        dist = Categorical(logits=logits)

        mask = jnp.array([True, False, True])
        masked_dist = dist.mask(mask)

        assert jnp.isfinite(masked_dist.logits[0])
        assert jnp.isneginf(masked_dist.logits[1])
        assert jnp.isfinite(masked_dist.logits[2])

    def test_mask_affects_mode(self):
        """Test that masking affects the mode."""
        logits = jnp.array([1.0, 10.0, 2.0])
        dist = Categorical(logits=logits)

        assert dist.mode() == 1

        mask = jnp.array([True, False, True])
        masked_dist = dist.mask(mask)

        assert masked_dist.mode() == 2

    def test_mask_forces_valid_samples(self):
        """Test that masking forces samples to be from valid categories."""
        logits = jnp.array([1.0, 1.0, 1.0, 1.0])
        dist = Categorical(logits=logits)

        mask = jnp.array([True, False, False, True])
        masked_dist = dist.mask(mask)

        keys = jr.split(jr.key(0), 50)
        for k in keys:
            sample = masked_dist.sample(k)
            assert sample in [0, 3]

    def test_batched_operations(self):
        """Test that operations work with batched inputs."""
        logits = jnp.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
        dist = Categorical(logits=logits)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (2,)
