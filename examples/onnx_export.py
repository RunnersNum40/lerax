import numpy as np
import onnxruntime as ort
from jax import random as jr

from lerax.env.classic_control import CartPole
from lerax.export import to_onnx
from lerax.policy import MLPActorCriticPolicy

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=jr.key(0))

to_onnx(policy, output_path="policy.onnx")

session = ort.InferenceSession("policy.onnx")
input_name = session.get_inputs()[0].name

observation = np.zeros(env.observation_space.flat_size, dtype=np.float32)
action = session.run(None, {input_name: observation})[0]

print(f"Observation: {observation}")
print(f"Action: {action}")
