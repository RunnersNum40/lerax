from __future__ import annotations

import pytest

from oryx.algorithm import AbstractAlgorithm


class TestAbstractAlgorithm:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractAlgorithm()  # pyright: ignore
