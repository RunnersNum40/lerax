import equinox as eqx
import jax.numpy as jnp

from lerax.target import HardUpdate


class Params(eqx.Module):
    weight: jnp.ndarray
    label: str


def test_hard_update_returns_online_exactly():
    online = Params(weight=jnp.array([1.0, 2.0, 3.0]), label="online")
    target = Params(weight=jnp.array([-1.0, -2.0, -3.0]), label="target")

    updated = HardUpdate()(online, target)

    assert jnp.array_equal(updated.weight, online.weight)
    assert updated.label == "online"


def test_hard_update_ignores_target_contents():
    online = Params(weight=jnp.zeros(3), label="online")
    different_targets = [
        Params(weight=jnp.ones(3), label="target-a"),
        Params(weight=jnp.full(3, -5.0), label="target-b"),
    ]

    for target in different_targets:
        updated = HardUpdate()(online, target)
        assert jnp.array_equal(updated.weight, online.weight)
