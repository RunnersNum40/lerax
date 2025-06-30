from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, Key

from oryx.distributions import AbstractDistribution
from oryx.spaces import AbstractSpace

from .base_policy import AbstractPolicy


class AbstractActorCriticPolicy(AbstractPolicy, strict=True):
    """
    Abstract base class for actor-critic algorithm policies.

    Has both a value and action prediction.
    """

    observation_space: eqx.AbstractVar[AbstractSpace]
    action_space: eqx.AbstractVar[AbstractSpace]

    @abstractmethod
    def extract_features(
        self, state: eqx.nn.State, observation, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, Float[Array, " features"]]:
        """Preprocess an observation into a set of features in a latent space."""

    @abstractmethod
    def value_from_features(
        self,
        state: eqx.nn.State,
        features: Float[Array, " features"],
        *,
        key: Key | None = None,
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        pass

    @abstractmethod
    def action_dist_from_features(
        self,
        state: eqx.nn.State,
        features: Float[Array, " features"],
        *,
        key: Key | None = None,
    ) -> tuple[eqx.nn.State, AbstractDistribution]:
        pass

    def __call__(
        self, state: eqx.nn.State, observation, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, Array, Float[Array, ""], Float[Array, ""]]:
        """Predict the value and action from an observation."""
        state, features = self.extract_features(state, observation, key=key)
        state, value = self.value_from_features(state, features, key=key)
        state, action_dist = self.action_dist_from_features(state, features, key=key)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        log_prob = action_dist.log_prob(action).squeeze()

        return state, action, value, log_prob

    def predict(
        self, state: eqx.nn.State, observation, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, Float]:
        """Predict the action from an observation."""
        state, features = self.extract_features(state, observation, key=key)
        state, action_dist = self.action_dist_from_features(state, features, key=key)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return state, action

    def predict_value(
        self, state: eqx.nn.State, observation, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        """Predict the value from an observation."""
        state, features = self.extract_features(state, observation, key=key)
        return self.value_from_features(state, features, key=key)
