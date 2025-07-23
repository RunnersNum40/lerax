from __future__ import annotations

from typing import Callable

from jaxtyping import Float, Key
from tensorboardX import SummaryWriter

from oryx.buffers import RolloutBuffer
from oryx.env import AbstractEnvLike
from oryx.policies import AbstractActorCriticPolicy

from .on_policy import AbstractOnPolicyAlgorithm


class PPO[ActType, ObsType](AbstractOnPolicyAlgorithm[ActType, ObsType]):
    """Proximal Policy Optimization (PPO) algorithm."""

    env: AbstractEnvLike[ActType, ObsType]
    policy: AbstractActorCriticPolicy[Float, ActType, ObsType]

    def __init__(
        self,
        env: AbstractEnvLike[ActType, ObsType],
        policy: AbstractActorCriticPolicy[Float, ActType, ObsType],
    ):
        self.env = env
        self.policy = policy

    def learn(
        self,
        callback: Callable | None = None,
        *,
        key: Key | None = None,
        progress_bar: bool = False,
        tb_log_name: str | None = None,
        log_interval: int = 100,
    ) -> PPO[ActType, ObsType]:
        raise NotImplementedError

    def train(
        self,
        rollout_buffer: RolloutBuffer[ActType, ObsType],
        *,
        key: Key,
        tb_writer: SummaryWriter | None = None,
    ) -> PPO[ActType, ObsType]:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> PPO[ActType, ObsType]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError
