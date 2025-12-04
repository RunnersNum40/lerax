from __future__ import annotations

from typing import ClassVar

from jaxtyping import Array, Integer, Key, Real

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.space import AbstractSpace, Discrete

from ..q import MLPQPolicy
from .base_dqn import AbstractStatelessDQNPolicy


class MLPDQNPolicy[ObsType: Real[Array, "..."]](
    AbstractStatelessDQNPolicy[ObsType, MLPQPolicy]
):
    """
    Deep Q-Network (DQN) policy with MLP Q-network components.

    Attributes:
        name: Name of the policy class.
        action_space: The action space of the environment.
        observation_space: The observation space of the environment.
        q_network: The Q-network used for action selection.
        target_q_network: The target Q-network used for stable learning.

    Args:
        env: The environment to create the policy for.
        epsilon: The epsilon value for epsilon-greedy action selection.
        width_size: The width of the hidden layers in the MLP.
        depth: The number of hidden layers in the MLP.
        key: JAX PRNG key for parameter initialization.

    Raises:
        ValueError: If the environment's action space is not Discrete.
    """

    name: ClassVar[str] = "MLPDQNPolicy"

    action_space: Discrete
    observation_space: AbstractSpace[ObsType]

    q_network: MLPQPolicy
    target_q_network: MLPQPolicy

    def __init__[StateType: AbstractEnvLikeState](
        self,
        env: AbstractEnvLike[StateType, Integer[Array, ""], ObsType],
        *,
        epsilon: float = 0.1,
        width_size: int = 64,
        depth: int = 2,
        key: Key,
    ):
        if not isinstance(env.action_space, Discrete):
            raise ValueError(
                f"MLPQPolicy only supports Discrete action spaces, got {type(env.action_space)}"
            )

        self.q_network = MLPQPolicy(
            env, epsilon=epsilon, width_size=width_size, depth=depth, key=key
        )
        self.target_q_network = MLPQPolicy(
            env, epsilon=epsilon, width_size=width_size, depth=depth, key=key
        )
        self.action_space = env.action_space
        self.observation_space = env.observation_space
