from __future__ import annotations

import pytest
from distreqx import bijectors
from jax import numpy as jnp
from jax import random as jr

from oryx.distribution import (
    Bernoulli,
    Categorical,
    MultivariateNormalDiag,
    Normal,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)


def test_bernoulli():
    key = jr.key(0)
    probabilities = jnp.array([0.2, 0.5, 0.8])
    distribution = Bernoulli(probs=probabilities)

    sample = distribution.sample(key)
    assert sample.shape == probabilities.shape
    assert jnp.all(jnp.logical_or(sample == 0, sample == 1))

    log_prob_vector = distribution.log_prob(sample)
    assert log_prob_vector.shape == probabilities.shape
    expected_lp = sample * jnp.log(probabilities) + (1 - sample) * jnp.log(
        1 - probabilities
    )
    assert jnp.allclose(log_prob_vector, expected_lp, atol=1e-6)

    expected_entropy = -(
        probabilities * jnp.log(probabilities)
        + (1 - probabilities) * jnp.log(1 - probabilities)
    )
    assert jnp.allclose(distribution.entropy(), expected_entropy, atol=1e-6)


def test_categorical():
    key = jr.key(0)
    class_logits = jnp.zeros(4)
    distribution = Categorical(logits=class_logits)

    sample = distribution.sample(key)
    assert int(sample) in range(4)
    assert jnp.allclose(distribution.probs, jnp.full((4,), 0.25))

    lp = distribution.log_prob(sample)
    assert lp.shape == ()
    assert jnp.allclose(lp, jnp.log(0.25))

    assert jnp.allclose(distribution.entropy(), jnp.log(4.0))


def test_normal():
    key = jr.key(0)
    location = jnp.array([0.0, 1.0])
    scale = jnp.array([1.0, 2.0])
    distribution = Normal(loc=location, scale=scale)

    sample = distribution.sample(key)
    assert sample.shape == location.shape
    assert jnp.allclose(distribution.mean(), location)
    assert jnp.all(
        distribution.log_prob(location) > distribution.log_prob(location + 5.0 * scale)
    )

    expected_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.square(scale))
    assert jnp.allclose(distribution.entropy(), expected_entropy, atol=1e-6)


def test_squashed_normal():
    key = jr.key(0)
    location = jnp.array([0.0, 0.5, -0.5])
    scale = jnp.array([0.3, 0.2, 0.4])
    distribution = SquashedNormal(loc=location, scale=scale)

    sample = distribution.sample(key)
    assert sample.shape == location.shape
    assert jnp.all(sample > -1.0) and jnp.all(sample < 1.0)
    assert isinstance(distribution.bijector, bijectors.Tanh)

    sample2, log_prob2 = distribution.sample_and_log_prob(key)
    assert jnp.allclose(distribution.log_prob(sample2), log_prob2, atol=1e-6)

    with pytest.raises(NotImplementedError):
        distribution.entropy()


def test_mvn_diag():
    key = jr.key(0)
    location = jnp.array([0.0, 1.0, -1.0])
    scale_diag = jnp.array([0.5, 0.3, 0.7])
    distribution = MultivariateNormalDiag(loc=location, scale_diag=scale_diag)

    sample = distribution.sample(key)
    assert sample.shape == location.shape
    assert jnp.allclose(distribution.mean(), location)

    sample2, log_prob2 = distribution.sample_and_log_prob(key)
    assert sample2.shape == location.shape
    assert log_prob2.shape == ()
    assert jnp.allclose(distribution.log_prob(sample2), log_prob2, atol=1e-6)

    assert distribution.log_prob(location) > distribution.log_prob(
        location + 5.0 * scale_diag
    )

    d = location.size
    expected_entropy = 0.5 * d * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(
        jnp.log(scale_diag)
    )
    assert jnp.allclose(distribution.entropy(), expected_entropy, atol=1e-6)


def test_squashed_mvn_diag():
    tanh_key, box_key = jr.split(jr.key(0))
    location = jnp.array([0.0, 0.5])
    scale_diag = jnp.array([0.2, 0.3])

    tanh_distribution = SquashedMultivariateNormalDiag(
        loc=location, scale_diag=scale_diag
    )
    tanh_sample = tanh_distribution.sample(tanh_key)
    assert jnp.all(tanh_sample > -1.0) and jnp.all(tanh_sample < 1.0)
    assert isinstance(tanh_distribution.bijector, bijectors.Tanh)
    s2, lp2 = tanh_distribution.sample_and_log_prob(tanh_key)
    assert jnp.allclose(tanh_distribution.log_prob(s2), lp2, atol=1e-6)

    with pytest.raises(NotImplementedError):
        tanh_distribution.entropy()

    bounded_low = jnp.array([-2.0, 0.0])
    bounded_high = jnp.array([2.0, 3.0])
    bounded_distribution = SquashedMultivariateNormalDiag(
        loc=location, scale_diag=scale_diag, low=bounded_low, high=bounded_high
    )
    bounded_sample = bounded_distribution.sample(box_key)
    assert jnp.all(bounded_sample > bounded_low) and jnp.all(
        bounded_sample < bounded_high
    )
    assert isinstance(bounded_distribution.bijector, bijectors.Chain)
    s3, lp3 = bounded_distribution.sample_and_log_prob(box_key)
    assert jnp.allclose(bounded_distribution.log_prob(s3), lp3, atol=1e-4)

    with pytest.raises(NotImplementedError):
        bounded_distribution.entropy()
