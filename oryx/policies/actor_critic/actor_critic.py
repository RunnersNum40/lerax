from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, Key

from oryx.distributions import AbstractDistribution

from ..base_policy import AbstractPolicy


# TODO: Break out stateful and non-stateful policies
class AbstractActorCriticPolicy[FeatureType, ActType, ObsType](
    AbstractPolicy[ActType, ObsType]
):
    """
    Base class for actor-critic policies.

    For now all policies are treated as stateful, meaning they maintain an internal
    state. Non-stateful policies can simplly ignore the state during implementation.
    """

    state_index: eqx.AbstractVar[eqx.nn.StateIndex]

    @abstractmethod
    def extract_features(
        self, state: eqx.nn.State, observation: ObsType
    ) -> tuple[eqx.nn.State, FeatureType]:
        """Extract features from an observation."""

    @abstractmethod
    def action_dist_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[
        eqx.nn.State,
        AbstractDistribution[ActType],
    ]:
        """Return an action distribution from features."""

    @abstractmethod
    def value_from_features(
        self, state: eqx.nn.State, features: FeatureType
    ) -> tuple[eqx.nn.State, Float]:
        """Return a value from features."""

    def __call__(
        self, state: eqx.nn.State, observation: ObsType, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, ActType, Float[Array, ""], Float[Array, ""]]:
        """
        Get an action and value from an observation.

        If `key` is provided, it will be used for sampling actions, if no key is
        provided the policy will return the most likely action.
        """
        state, features = self.extract_features(state, observation)
        state, action_dist = self.action_dist_from_features(state, features)
        state, value = self.value_from_features(state, features)

        if key is None:
            action = action_dist.mode()
            log_prob = action_dist.log_prob(action)
        else:
            action, log_prob = action_dist.sample_and_log_prob(key)

        return state, action, value, log_prob.squeeze()

    def predict(
        self, state: eqx.nn.State, observation: ObsType, *, key: Key | None = None
    ) -> tuple[eqx.nn.State, ActType]:
        """Get an action from an observation."""
        state, features = self.extract_features(state, observation)
        state, action_dist = self.action_dist_from_features(state, features)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return state, action

    def value(
        self, state: eqx.nn.State, observation: ObsType
    ) -> tuple[eqx.nn.State, Float[Array, ""]]:
        """Get the value of an observation."""
        state, features = self.extract_features(state, observation)
        return self.value_from_features(state, features)

    @abstractmethod
    def reset(self, state: eqx.nn.State) -> eqx.nn.State:
        """Reset the policy state."""
