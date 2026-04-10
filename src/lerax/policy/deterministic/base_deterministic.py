from __future__ import annotations

from typing import Any

import equinox as eqx

from lerax.space import AbstractSpace

from ..base_policy import AbstractPolicy, AbstractPolicyState


class AbstractDeterministicPolicy[
    StateType: AbstractPolicyState | None,
    ActType,
    ObsType,
](AbstractPolicy[StateType, ActType, ObsType, None]):
    """
    Base class for deterministic policies used by DDPG and TD3.

    Deterministic policies output a single action for each observation
    (no sampling). Exploration noise is added by the algorithm, not the
    policy. When ``key`` is provided to ``__call__``, the policy may
    optionally add noise for exploration; when ``key`` is ``None``, the
    policy must return the deterministic action.

    Attributes:
        name: The name of the policy.
        action_space: The action space of the policy.
        observation_space: The observation space of the policy.
    """

    name: eqx.AbstractClassVar[str]
    action_space: eqx.AbstractVar[AbstractSpace[ActType, None]]
    observation_space: eqx.AbstractVar[AbstractSpace[ObsType, Any]]
