from typing import Any

import equinox as eqx
import jax
import numpy as np
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from matplotlib import pyplot as plt

from lerax.model import MLPNeuralCDE, NCDEState
from lerax.utils import debug_wrapper, filter_scan


def random_matrix(n: int, min_eig: float, max_eig: float, *, key: Any) -> Any:
    key, eig_key, matrix_key = jr.split(key, 3)
    eigvals = jr.uniform(eig_key, (n,), minval=min_eig, maxval=max_eig)
    rand_mat = jr.normal(matrix_key, (n, n))
    Q, R = jnp.linalg.qr(rand_mat)
    diag = jnp.sign(jnp.diag(R))
    Q = Q * diag
    A = Q @ jnp.diag(eigvals) @ Q.T
    return A


def get_data(
    n: int,
    length: int,
    t0: float,
    t1: float,
    *,
    add_noise: bool = True,
    noise_scale: float = 1e-3,
    key: Any,
):
    initial_key, matrix_key, noise_key = jr.split(key, 3)
    ts = jnp.linspace(t0, t1, length)
    x0 = jr.uniform(initial_key, (n,))
    matrix = random_matrix(n, -1, 1e-1, key=matrix_key)
    xs = jax.vmap(lambda t: jsp.linalg.expm(t * matrix) @ x0)(ts)
    if add_noise:
        xs = xs + jr.normal(noise_key, xs.shape) * noise_scale
    y1 = jnp.diag(matrix)
    return xs, ts, y1


def simulate_trajectory(state: NCDEState, model: MLPNeuralCDE, ts, xs):
    state, _ = lax.scan(lambda c, i: (model(c, *i)[0], None), state, (ts, xs))
    return state, model.y1(state)


@eqx.filter_value_and_grad
def loss_grad(neural_cde: MLPNeuralCDE, state: NCDEState, xs_batch, ts_batch, y_batch):
    y_preds = jax.vmap(
        lambda ts, xs: simulate_trajectory(state, neural_cde, ts, xs)[1]
    )(ts_batch, xs_batch)
    return jnp.mean((y_preds - y_batch) ** 2)


dataset_size = 256
data_length = 128
data_time = 4 * jnp.pi
add_noise = True
data_dim = 2

learning_rate = 3e-4
epochs = 1024
batch_size = 32
seed = 0
test_size = max(256, dataset_size // 4)

key = jr.key(seed)
key, data_key, model_key = jr.split(key, 3)

xs, ts, y1 = jax.vmap(
    lambda k, t0: get_data(
        data_dim, data_length, t0=t0, t1=t0 + data_time, add_noise=add_noise, key=k
    )
)(jr.split(data_key, dataset_size), jnp.linspace(0, 1, dataset_size))

neural_cde = MLPNeuralCDE(
    in_size=data_dim,
    out_size=data_dim,
    latent_size=2,
    output_depth=0,
    key=model_key,
)
state = neural_cde.reset()
schedule = optax.cosine_decay_schedule(
    learning_rate, decay_steps=(dataset_size // batch_size) * epochs
)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(eqx.filter(neural_cde, eqx.is_inexact_array))

num_batches = dataset_size // batch_size


def batch_scan(carry, inputs):
    state, model, opt_state = carry
    xs_b, ts_b, y_b = inputs
    loss, grads = loss_grad(model, state, xs_b, ts_b, y_b)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return (state, model, opt_state), loss


def epoch_scan(carry, i):
    state, model, opt_state, key = carry
    carry_key, batch_key = jr.split(key)
    perm = jr.permutation(batch_key, dataset_size)
    idx = perm.reshape(num_batches, -1)
    (state, model, opt_state), losses = filter_scan(
        batch_scan, (state, model, opt_state), (xs[idx], ts[idx], y1[idx])
    )
    debug_wrapper(print)("Epoch", i)
    return (state, model, opt_state, carry_key), jnp.mean(losses)


(state, neural_cde, opt_state, key), epoch_losses = filter_scan(
    epoch_scan, (state, neural_cde, opt_state, key), xs=jnp.arange(epochs)
)

final_train_loss = float(epoch_losses[-1])
print(f"Final training loss: {final_train_loss:.6f}")

plt.figure(figsize=(6, 4))
plt.plot(np.arange(epochs), np.asarray(epoch_losses))
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Training loss")
plt.tight_layout()
plt.show()

key, test_key = jr.split(key)
test_xs, test_ts, test_y1 = jax.vmap(
    lambda k, t0: get_data(
        data_dim, data_length, t0=t0, t1=t0 + data_time, add_noise=add_noise, key=k
    )
)(jr.split(test_key, test_size), jnp.linspace(0, 1, test_size))

y_pred_test = jax.vmap(
    lambda ts_i, xs_i: simulate_trajectory(state, neural_cde, ts_i, xs_i)[1]
)(test_ts, test_xs)

test_mse = float(jnp.mean((y_pred_test - test_y1) ** 2))
print(f"Test MSE: {test_mse:.6f}")

state = neural_cde.reset()
idx = 0

ts = test_ts[idx]
xs = test_xs[idx]
y_true = test_y1[idx]
y_pred = y_pred_test[idx]

state, ys = simulate_trajectory(state, neural_cde, ts, xs)

z0 = neural_cde.z0(ts[0], xs[0])
zs = neural_cde.zs(state)
ys = neural_cde.ys(state)

fig, (ax_latent, ax_out) = plt.subplots(1, 2, figsize=(8, 4))

z = np.asarray(zs)
ax_latent.plot(z[:, 0], z[:, 1], lw=1)
ax_latent.scatter(z[0, 0], z[0, 1], s=25, label="start")
ax_latent.scatter(z[-1, 0], z[-1, 1], s=25, label="end")
ax_latent.set_xlabel("z[0]")
ax_latent.set_ylabel("z[1]")
ax_latent.set_title("Latent trajectory z(t)")
ax_latent.set_aspect("equal", adjustable="box")
ax_latent.legend(loc="best")

y_path = np.asarray(ys)
yt = np.asarray(y_true)
yp = np.asarray(y_pred)
ax_out.plot(y_path[:, 0], y_path[:, 1], lw=1)
ax_out.scatter(yt[0], yt[1], marker="x", s=48, label="true")
ax_out.scatter(yp[0], yp[1], marker="o", s=36, label="pred")
ax_out.set_xlabel("y[0]")
ax_out.set_ylabel("y[1]")
ax_out.set_title("Output path with true/pred")
ax_out.set_aspect("equal", adjustable="box")
ax_out.legend(loc="best")

fig.tight_layout()
plt.show()
