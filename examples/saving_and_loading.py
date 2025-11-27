from jax import random as jr

from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)

policy.serialize("model.eqx")  # Save with fixed name
policy.serialize("model_epoch_{epoch}.eqx", epoch=10)  # Save with formatted name
policy.serialize(
    lambda epoch: f"model_epoch_{epoch}.eqx", epoch=10  # Save with dynamic name
)
policy_loaded = MLPActorCriticPolicy.deserialize("model.eqx", env, key=jr.key(1))
