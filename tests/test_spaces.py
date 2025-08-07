import jax.numpy as jnp
from jax import random as jr

from oryx.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    OneOf,
    Tuple,
    flatten,
)


def test_discrete():
    key = jr.key(0)
    a = Discrete(5, start=2)
    b = Discrete(5, start=2)
    c = Discrete(4, start=0)

    sample = a.sample(key)
    assert 2 <= int(sample) < 7, "sample should lie in [start, start + n)"
    assert a.contains(3), "contains must accept in-range values"
    assert not a.contains(1), "contains must reject out-of-range values"
    assert a == b, "identical parameterisation => equality"
    assert a != c, "different parameterisation => inequality"


def test_box():
    key = jr.key(0)
    a = Box(low=-1.0, high=1.0, shape=(2,))
    b = Box(low=-jnp.ones(2), high=jnp.ones(2), shape=(2,))
    c = Box(low=0.0, high=1.0, shape=(2,))

    sample = a.sample(key)
    assert jnp.all(sample >= a.low) and jnp.all(
        sample <= a.high
    ), "sample must fall inside bounds"

    midpoint = jnp.zeros_like(a.low)
    assert a.contains(midpoint), "midpoint expected to be inside Box"
    assert not a.contains(a.high + 0.1), "value above high bound must be rejected"

    assert a == b, "broadcasted parameters with same values => equality"
    assert a != c, "different bounds => inequality"


def test_tuple():
    key = jr.key(0)
    spaces = (Discrete(3), Box(-1, 1, shape=()))
    a = Tuple(spaces)
    b = Tuple(spaces)
    c = Tuple((Discrete(2), spaces[1]))

    sample = a.sample(key)
    assert isinstance(sample, tuple) and len(sample) == 2, "sample must be a 2-tuple"
    assert a.contains(sample), "contains should accept own sample"
    assert not a.contains((10, 0.0)), "invalid tuple should be rejected"

    assert a == b, "identical Tuple spaces => equality"
    assert a != c, "different sub-space => inequality"


def test_dict():
    key = jr.key(0)
    spec = {"a": Discrete(2), "b": Box(0, 1, shape=())}
    a = Dict(spec)
    b = Dict(spec)
    c = Dict({"a": Discrete(2), "b": Box(0, 2, shape=())})

    sample = a.sample(key)
    assert set(sample.keys()) == set(spec), "sample keys must match spec"
    assert a.contains(sample), "contains should accept own sample"
    assert not a.contains({"a": -1, "b": 0.5}), "invalid dict should be rejected"

    assert a == b, "identical Dict spaces => equality"
    assert a != c, "different sub-space => inequality"


def test_multidiscrete():
    key = jr.key(0)
    a = MultiDiscrete((2, 3), starts=(1, 2))
    b = MultiDiscrete((2, 3), starts=(1, 2))
    c = MultiDiscrete((2, 3), starts=(0, 0))

    sample = a.sample(key)
    assert sample.shape == (2,), "sample shape must equal (n_dims,)"
    assert a.contains(sample), "contains should accept own sample"
    assert not a.contains(jnp.array([0, 5])), "out-of-range array must be rejected"

    assert a == b, "identical parameters => equality"
    assert a != c, "different starts => inequality"


def test_multibinary():
    key = jr.key(0)
    a = MultiBinary(4)
    b = MultiBinary(4)
    c = MultiBinary(3)

    sample = a.sample(key)
    assert jnp.all(jnp.logical_or(sample == 0, sample == 1)), "sample must be binary"
    assert a.contains(sample), "contains should accept own sample"
    assert not a.contains(jnp.array([0, 1, 2, 1])), "non-binary entry must be rejected"

    assert a == b, "identical n => equality"
    assert a != c, "different n => inequality"


def test_oneof():
    key = jr.key(0)
    sub_a, sub_b = Discrete(3), Box(-1, 1, shape=())
    a = OneOf((sub_a, sub_b))
    b = OneOf((sub_a, sub_b))
    c = OneOf((Discrete(4), sub_b))

    sample = a.sample(key)
    assert sub_a.contains(sample) or sub_b.contains(
        sample
    ), "sample from OneOf must belong to at least one sub-space"
    assert a.contains(sample), "contains should accept own sample"
    assert not a.contains("invalid"), "unrelated type must be rejected"

    assert a == b, "same sub-spaces (order-agnostic) => equality"
    assert a != c, "different sub-spaces => inequality"


def test_flatten_exhaustive():
    key = jr.key(0)

    box_space = Box(-1.0, 1.0, shape=(2, 3))
    key, sample_key = jr.split(key)
    box_sample = box_space.sample(sample_key)
    flat_box = flatten(box_space, box_sample)
    assert flat_box.ndim == 1
    assert jnp.array_equal(flat_box, box_sample.ravel())

    discrete_space = Discrete(5, start=2)
    discrete_sample = jnp.asarray(3)
    flat_disc = flatten(discrete_space, discrete_sample)
    assert flat_disc.shape == (1,)
    assert jnp.array_equal(flat_disc, jnp.asarray([discrete_sample]))

    mb_space = MultiBinary(4)
    key, sample_key = jr.split(key)
    mb_sample = mb_space.sample(sample_key)
    flat_mb = flatten(mb_space, mb_sample)
    assert jnp.array_equal(flat_mb, mb_sample.ravel())

    md_space = MultiDiscrete((2, 3), starts=(0, 1))
    md_sample = jnp.asarray([1, 2])
    flat_md = flatten(md_space, md_sample)
    assert jnp.array_equal(flat_md, md_sample.ravel())

    tuple_space = Tuple((box_space, discrete_space, mb_space))
    tuple_sample = (box_sample, discrete_sample, mb_sample)
    expected_tuple = jnp.concatenate(
        [box_sample.ravel(), jnp.asarray([discrete_sample]), mb_sample.ravel()]
    )
    flat_tuple = flatten(tuple_space, tuple_sample)
    assert jnp.array_equal(flat_tuple, expected_tuple)

    dict_space = Dict({"b": box_space, "a": discrete_space})
    dict_sample = {"b": box_sample, "a": discrete_sample}
    expected_dict = jnp.concatenate(
        [jnp.asarray([discrete_sample]), box_sample.ravel()]
    )
    flat_dict = flatten(dict_space, dict_sample)
    assert jnp.array_equal(flat_dict, expected_dict)

    total_expected_len = (
        box_sample.size
        + 1
        + mb_sample.size
        + md_sample.size
        + expected_tuple.size
        + expected_dict.size
    )
    total_flat_len = (
        flat_box.size
        + flat_disc.size
        + flat_mb.size
        + flat_md.size
        + flat_tuple.size
        + flat_dict.size
    )
    assert total_flat_len == total_expected_len
