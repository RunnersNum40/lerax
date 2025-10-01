from __future__ import annotations

import pytest
from jax import numpy as jnp
from jax import random as jr

from oryx.space import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Tuple,
    flat_dim,
    flatten,
)


def test_discrete():
    key = jr.key(0)
    a = Discrete(5, start=2)
    b = Discrete(5, start=2)
    c = Discrete(4, start=0)

    sample = a.sample(key)
    assert 2 <= int(sample) < 7

    assert a.contains(3)
    assert a.contains(jnp.array(4))
    assert a.contains(jnp.array(2))

    assert not a.contains(1)
    assert not a.contains(jnp.array(10))
    assert not a.contains(3.5)
    assert not a.contains(jnp.array([3, 4]))
    assert not a.contains("string")

    assert a == b
    assert a != c


def test_box():
    key = jr.key(0)
    a = Box(low=-1.0, high=1.0, shape=(2,))
    b = Box(low=-jnp.ones(2), high=jnp.ones(2))
    c = Box(low=0.0, high=1.0, shape=(2,))

    sample = a.sample(key)
    assert jnp.all(sample >= a.low) and jnp.all(sample <= a.high)

    midpoint = jnp.zeros_like(a.low)
    assert a.contains(midpoint)
    assert a.contains(jnp.zeros(2))

    assert not a.contains(a.high + 0.1)
    assert not a.contains(a.low - 0.1)
    assert not a.contains(jnp.array([0.0, 2.0]))
    assert not a.contains(jnp.array([0.0]))
    assert not a.contains(0.0)
    assert not a.contains("string")

    assert a == b
    assert a != c


def test_tuple():
    key = jr.key(0)
    spaces = (Discrete(3), Box(-1, 1))
    a = Tuple(spaces)
    b = Tuple(spaces)
    c = Tuple((Discrete(2), spaces[1]))

    sample = a.sample(key)
    assert isinstance(sample, tuple) and len(sample) == 2
    assert a.contains(sample)
    assert a.contains((1, 0.0))

    assert not a.contains((10, 0.0))
    assert not a.contains((1, 2.0))
    assert not a.contains((1,))
    assert not a.contains([1, 0.0])
    assert not a.contains("string")

    assert a == b
    assert a != c


def test_dict():
    key = jr.key(0)
    spec = {"a": Discrete(2), "b": Box(0, 1, shape=())}
    a = Dict(spec)
    b = Dict(spec)
    c = Dict({"a": Discrete(2), "b": Box(0, 2, shape=())})

    sample = a.sample(key)
    assert set(sample.keys()) == set(spec)
    assert a.contains(sample)
    assert a.contains({"a": 1, "b": 0.5})

    assert not a.contains({"a": -1, "b": 0.5})
    assert not a.contains({"a": 1, "b": 2.0})
    assert not a.contains({"a": 1})
    assert not a.contains({"a": 1, "b": 0.5, "c": 3})
    assert not a.contains([1, 0.5])
    assert not a.contains("string")

    assert a == b
    assert a != c


def test_multidiscrete():
    key = jr.key(0)
    a = MultiDiscrete((2, 3), starts=(1, 2))
    b = MultiDiscrete((2, 3), starts=(1, 2))
    c = MultiDiscrete((2, 3), starts=(0, 0))

    sample = a.sample(key)
    print(sample)
    assert sample.shape == (2,)
    assert a.contains(sample)
    assert a.contains(jnp.array([1, 2]))

    assert not a.contains(jnp.array([0, 5]))
    assert not a.contains(jnp.array([1.5, 2]))
    assert not a.contains(jnp.array([1, 2, 1]))
    assert not a.contains("string")

    assert a == b
    assert a != c


def test_multibinary():
    key = jr.key(0)
    a = MultiBinary(4)
    b = MultiBinary(4)
    c = MultiBinary(3)

    sample = a.sample(key)
    assert jnp.all(jnp.logical_or(sample == 0, sample == 1))
    assert a.contains(sample)
    assert a.contains(jnp.array([0, 1, 1, 0]))

    assert not a.contains(jnp.array([0, 1, 2, 1]))
    assert not a.contains(jnp.array([0, 1, 1]))
    assert not a.contains(jnp.array([0.5, 1.0, 1.0, 0.0]))
    assert not a.contains("string")

    assert a == b
    assert a != c


def test_flatten():
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


@pytest.mark.parametrize(
    "space",
    [
        Box(-1.0, 1.0, shape=(2, 3)),
        Discrete(7, start=0),
        MultiBinary(5),
        MultiDiscrete((2, 3, 4), starts=(0, 1, 0)),
        Tuple(
            (
                Box(0.0, 1.0, shape=(2,)),
                Discrete(3),
                MultiBinary(2),
            )
        ),
        Dict(
            {
                "b": Box(-1.0, 1.0, shape=(2,)),
                "a": Discrete(4),
            }
        ),
    ],
)
def test_flat_dim(space):
    key = jr.key(0)
    sample = space.sample(key)
    flat = flatten(space, sample)
    assert flat.ndim == 1
    assert flat.size == flat_dim(space)
