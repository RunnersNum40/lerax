import equinox as eqx
import pytest

from oryx.model.base_model import AbstractModel, AbstractStatefulModel


class TestAbstractModel:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractModel()  # pyright: ignore

    def test_missing_call_keeps_abstract(self):
        class NoCall(AbstractModel):  # type: ignore
            pass

        with pytest.raises(TypeError):
            NoCall()  # pyright: ignore


class TestAbstractStatefulModel:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractStatefulModel()  # pyright: ignore

    def test_missing_call_keeps_abstract(self):
        class NoCall(AbstractStatefulModel):  # type: ignore
            state_index: eqx.nn.StateIndex[None] = eqx.nn.StateIndex(None)

        with pytest.raises(TypeError):
            NoCall()  # pyright: ignore

    def test_state_roundtrip(self):
        class Stateful(AbstractStatefulModel):
            state_index: eqx.nn.StateIndex[None] = eqx.nn.StateIndex(None)

            def __call__(self, state):
                return state, None

            def reset(self, state):
                return state

        model, state = eqx.nn.make_with_state(Stateful)()
        state, _ = model(state)
