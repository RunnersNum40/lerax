from pathlib import Path
from tempfile import TemporaryDirectory

import jax
from jax import random as jr

from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy


def test_saving_and_loading():
    """Test policy serialization and deserialization."""
    policy_key = jr.key(0)

    env = CartPole()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.eqx"

        policy.serialize(str(model_path))
        assert model_path.exists()

        policy_loaded = MLPActorCriticPolicy.deserialize(
            str(model_path), env=env, key=jr.key(1)
        )

        assert policy_loaded is not None

        # Verify the loaded policy has the same structure
        original_leaves = jax.tree.leaves(policy)
        loaded_leaves = jax.tree.leaves(policy_loaded)
        assert len(original_leaves) == len(loaded_leaves)
