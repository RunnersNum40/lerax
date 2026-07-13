import equinox as eqx
import jax.numpy as jnp
import pytest

from lerax.advantage import GAE, BootstrappedReturn, NStepReturn


@pytest.mark.parametrize(
    "estimator,dones",
    [
        (BootstrappedReturn(), jnp.array([False, False])),
        (NStepReturn(n=2), jnp.array([False, False])),
    ],
)
def test_common_estimators_are_jittable(estimator, dones):
    advantages, returns = eqx.filter_jit(estimator)(
        jnp.ones(2),
        jnp.zeros(2),
        dones,
        jnp.array(0.5),
    )

    assert advantages.shape == (2,)
    assert returns.shape == (2,)
    assert jnp.all(jnp.isfinite(advantages))
    assert jnp.all(jnp.isfinite(returns))


def test_concrete_estimators_do_not_inherit_from_each_other():
    estimator_types = (GAE, BootstrappedReturn, NStepReturn)

    for estimator_type in estimator_types:
        for other_type in estimator_types:
            if estimator_type is not other_type:
                assert not issubclass(estimator_type, other_type)
