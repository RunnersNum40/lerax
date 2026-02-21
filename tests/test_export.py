from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from jax import random as jr

from lerax.env.classic_control import CartPole, Pendulum
from lerax.export import to_onnx
from lerax.policy import MLPActorCriticPolicy, MLPQPolicy, MLPSACPolicy


def _run_onnx(proto: onnx.ModelProto, observation: np.ndarray) -> np.ndarray:
    """Run an ONNX model on an observation and return the output."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        onnx.save(proto, str(path))
        session = ort.InferenceSession(str(path))
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: observation})[0]
        assert isinstance(result, np.ndarray)
        return result


def test_actor_critic_export():
    """MLPActorCriticPolicy exports to a valid ONNX proto."""
    policy = MLPActorCriticPolicy(env=CartPole(), key=jr.key(0))
    proto = to_onnx(policy)
    assert isinstance(proto, onnx.ModelProto)


def test_q_policy_export():
    """MLPQPolicy exports to a valid ONNX proto."""
    policy = MLPQPolicy(env=CartPole(), key=jr.key(0))
    proto = to_onnx(policy)
    assert isinstance(proto, onnx.ModelProto)


def test_sac_policy_export():
    """MLPSACPolicy exports to a valid ONNX proto."""
    policy = MLPSACPolicy(env=Pendulum(), key=jr.key(0))
    proto = to_onnx(policy)
    assert isinstance(proto, onnx.ModelProto)


def test_export_to_file():
    """to_onnx writes a valid .onnx file when output_path is given."""
    policy = MLPActorCriticPolicy(env=CartPole(), key=jr.key(0))

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "policy.onnx"
        proto = to_onnx(policy, output_path=path)
        assert path.exists()
        assert path.stat().st_size > 0
        assert isinstance(proto, onnx.ModelProto)


def test_actor_critic_inference_matches():
    """ONNX output matches JAX for MLPActorCriticPolicy."""
    policy = MLPActorCriticPolicy(env=CartPole(), key=jr.key(0))
    proto = to_onnx(policy)

    obs = jnp.zeros(policy.observation_space.flat_size)
    _, jax_action = policy(None, obs)

    onnx_action = _run_onnx(proto, np.array(obs, dtype=np.float32))
    np.testing.assert_allclose(np.array(jax_action), onnx_action, atol=1e-5)


def test_q_policy_inference_matches():
    """ONNX output matches JAX for MLPQPolicy."""
    policy = MLPQPolicy(env=CartPole(), key=jr.key(0))
    proto = to_onnx(policy)

    obs = jnp.ones(policy.observation_space.flat_size)
    _, jax_action = policy(None, obs)

    onnx_action = _run_onnx(proto, np.array(obs, dtype=np.float32))
    np.testing.assert_allclose(np.array(jax_action), onnx_action, atol=1e-5)


def test_sac_policy_inference_matches():
    """ONNX output matches JAX for MLPSACPolicy."""
    policy = MLPSACPolicy(env=Pendulum(), key=jr.key(0))
    proto = to_onnx(policy)

    obs = jnp.ones(policy.observation_space.flat_size)
    _, jax_action = policy(None, obs)

    onnx_action = _run_onnx(proto, np.array(obs, dtype=np.float32))
    np.testing.assert_allclose(np.array(jax_action), onnx_action, atol=1e-5)
