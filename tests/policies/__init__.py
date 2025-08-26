import equinox as eqx
from jax import numpy as jnp

from oryx.distribution import Normal
from oryx.policy.actor_critic import AbstractActorCriticPolicy
from oryx.space import Box
from tests.envs import EchoEnv


class ConstantPolicy(AbstractActorCriticPolicy):
    new_value: float
    logp: float
    entropy_val: float

    state_index: eqx.nn.StateIndex[None] = eqx.nn.StateIndex(None)
    action_space = Box(-jnp.inf, jnp.inf, shape=())
    observation_space = Box(-jnp.inf, jnp.inf, shape=())

    def __init__(self, new_value: float, logp: float = 0.0, entropy_val: float = 0.0):
        self.new_value = float(new_value)
        self.logp = float(logp)
        self.entropy_val = float(entropy_val)

    def extract_features(self, state, observation):
        return state, jnp.asarray(0.0)

    def action_dist_from_features(self, state, features):
        raise NotImplementedError

    def value_from_features(self, state, features):
        return state, jnp.asarray(self.new_value)

    def reset(self, state):
        return state

    def evaluate_action(self, state, observation, action):
        return (
            state,
            jnp.asarray(self.new_value),
            jnp.asarray(self.logp),
            jnp.asarray(self.entropy_val),
        )


class NormalPolicy(AbstractActorCriticPolicy):
    env: EchoEnv
    state_index: eqx.nn.StateIndex[None] = eqx.nn.StateIndex(None)

    def __init__(self, env: EchoEnv):
        self.env = env

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def extract_features(self, state, observation):
        return state, jnp.asarray(observation)

    def action_dist_from_features(self, state, features):
        return state, Normal(features, jnp.asarray(1.0))

    def value_from_features(self, state, features):
        return state, jnp.asarray(0.0)

    def reset(self, state):
        return state
