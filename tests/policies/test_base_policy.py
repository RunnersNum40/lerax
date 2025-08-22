from __future__ import annotations

import pytest

from oryx.policy import AbstractPolicy
from oryx.policy.actor_critic import AbstractActorCriticPolicy


class TestAbstractPolicy:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractPolicy()  # pyright: ignore


class TestAbstractActorCriticPolicy:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractActorCriticPolicy()  # pyright: ignore
