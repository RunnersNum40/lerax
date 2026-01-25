from jax import numpy as jnp
from jax import random as jr

from lerax.distribution import Bernoulli


class TestBernoulli:
    """Tests for the Bernoulli distribution."""

    def test_init_with_logits(self):
        """Test initialization with logits."""
        logits = jnp.array([0.0, 1.0, -1.0])
        dist = Bernoulli(logits=logits)

        assert dist.logits.shape == (3,)
        assert jnp.allclose(dist.logits, logits)

    def test_init_with_probs(self):
        """Test initialization with probabilities."""
        probs = jnp.array([0.5, 0.8, 0.2])
        dist = Bernoulli(probs=probs)

        assert dist.probs.shape == (3,)
        assert jnp.allclose(dist.probs, probs)

    def test_sample_shape(self):
        """Test that samples have the correct shape."""
        logits = jnp.array([0.0, 1.0, -1.0])
        dist = Bernoulli(logits=logits)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (3,)
        assert jnp.issubdtype(sample.dtype, jnp.integer)

    def test_sample_and_log_prob(self):
        """Test sample_and_log_prob returns consistent values."""
        logits = jnp.array([0.0, 1.0, -1.0])
        dist = Bernoulli(logits=logits)

        key = jr.key(0)
        sample, log_prob = dist.sample_and_log_prob(key)

        expected_log_prob = dist.log_prob(sample)
        assert jnp.allclose(log_prob, expected_log_prob)

    def test_log_prob(self):
        """Test log probability computation."""
        probs = jnp.array([0.5])
        dist = Bernoulli(probs=probs)

        log_prob_true = dist.log_prob(jnp.array([True]))
        log_prob_false = dist.log_prob(jnp.array([False]))

        assert jnp.allclose(log_prob_true, jnp.log(0.5))
        assert jnp.allclose(log_prob_false, jnp.log(0.5))

    def test_prob(self):
        """Test probability computation."""
        probs = jnp.array([0.7])
        dist = Bernoulli(probs=probs)

        prob_true = dist.prob(jnp.array([True]))
        prob_false = dist.prob(jnp.array([False]))

        assert jnp.allclose(prob_true, 0.7)
        assert jnp.allclose(prob_false, 0.3)

    def test_entropy(self):
        """Test entropy computation."""
        dist_max_entropy = Bernoulli(probs=jnp.array([0.5]))
        entropy = dist_max_entropy.entropy()

        assert jnp.allclose(entropy, jnp.log(2.0))

    def test_mean(self):
        """Test mean computation."""
        probs = jnp.array([0.3, 0.7])
        dist = Bernoulli(probs=probs)

        mean = dist.mean()
        assert jnp.allclose(mean, probs)

    def test_mode(self):
        """Test mode computation."""
        probs = jnp.array([0.3, 0.7])
        dist = Bernoulli(probs=probs)

        mode = dist.mode()
        expected = jnp.array([False, True])
        assert jnp.array_equal(mode, expected)

    def test_mask(self):
        """Test masking functionality."""
        logits = jnp.array([0.0, 1.0, 2.0])
        dist = Bernoulli(logits=logits)

        mask = jnp.array([True, False, True])
        masked_dist = dist.mask(mask)

        assert jnp.isfinite(masked_dist.logits[0])
        assert jnp.isneginf(masked_dist.logits[1])
        assert jnp.isfinite(masked_dist.logits[2])

    def test_mask_affects_sampling(self):
        """Test that masking affects sampling probabilities."""
        logits = jnp.array([10.0])
        dist = Bernoulli(logits=logits)

        mask = jnp.array([False])
        masked_dist = dist.mask(mask)

        assert masked_dist.probs[0] < 1e-6

    def test_batched_operations(self):
        """Test that operations work with batched inputs."""
        logits = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        dist = Bernoulli(logits=logits)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (2, 2)
