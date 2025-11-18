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
            raise TypeError(
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
