from __future__ import annotations

from typing import ClassVar

from jaxtyping import Array, Float, Integer, Key, Real

from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.model import MLP
from lerax.space import AbstractSpace, Discrete

from .base_q import AbstractStatelessQPolicy


class MLPQPolicy[ObsType: Real[Array, "..."]](AbstractStatelessQPolicy[ObsType]):
    name: ClassVar[str] = "MLPQPolicy"

    action_space: Discrete
    observation_space: AbstractSpace[ObsType]

    epsilon: float
    q_network: MLP

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

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.epsilon = epsilon
        self.q_network = MLP(
            in_size=self.observation_space.flat_size,
            out_size=self.action_space.n,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def q_values(self, observation: ObsType) -> Float[Array, " actions"]:
        flat_obs = self.observation_space.flatten_sample(observation)
        return self.q_network(flat_obs)
