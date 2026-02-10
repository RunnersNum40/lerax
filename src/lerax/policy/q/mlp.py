from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
from jaxtyping import Array, Float, Integer, Key, Real

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Discrete

from .base_q import AbstractQPolicy


class MLPQPolicy[ObsType: Real[Array, "..."]](AbstractQPolicy[None, ObsType]):
    """
    Q-learning policy with an MLP Q-network.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        epsilon: The epsilon value for epsilon-greedy action selection.
        q_network: The MLP Q-network used for action value estimation.

    Args:
        env: The environment to create the policy for.
        epsilon: The epsilon value for epsilon-greedy action selection.
        width_size: The width of the hidden layers in the MLP.
        depth: The number of hidden layers in the MLP.
        key: JAX PRNG key for parameter initialization.

    Raises:
        ValueError: If the environment's action space is not Discrete.
    """

    name: ClassVar[str] = "MLPQPolicy"

    action_space: Discrete
    observation_space: AbstractSpace[ObsType, Any]

    epsilon: float
    q_network: eqx.nn.MLP

    def __init__[StateType: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[StateType, Integer[Array, ""], ObsType, Any],
        *,
        epsilon: float = 0.1,
        width_size: int = 64,
        depth: int = 2,
        key: Key[Array, ""],
    ):
        if not isinstance(env.action_space, Discrete):
            raise ValueError(
                f"MLPQPolicy only supports Discrete action spaces, got {type(env.action_space)}"
            )

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.epsilon = epsilon
        self.q_network = eqx.nn.MLP(
            in_size=self.observation_space.flat_size,
            out_size=self.action_space.n,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def reset(self, *, key: Key[Array, ""]) -> None:
        return None

    def q_values(
        self, state: None, observation: ObsType
    ) -> tuple[None, Float[Array, " actions"]]:
        flat_obs = self.observation_space.flatten_sample(observation)
        return None, self.q_network(flat_obs)
