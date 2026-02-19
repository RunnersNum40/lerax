from jax import random as jr

from lerax.env.classic_control import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key, learn_key = jr.split(jr.key(0), 2)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)

policy.serialize("model.eqx")
policy_loaded = MLPActorCriticPolicy.deserialize("model.eqx", env, key=jr.key(1))
