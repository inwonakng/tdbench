# source code from https://github.com/google-research/google-research/tree/master/kip
import functools
from jax.example_libraries import optimizers
import jax
import jax.config
from jax.config import config as jax_config

jax_config.update(
    "jax_enable_x64", True
)  # for numerical stability, can disable if not an issue
from jax import numpy as jnp
from jax import scipy as sp
import numpy as np
from neural_tangents import stax

from .random_sample import random_sample


def FullyConnectedNetwork(
    hidden_dims,
    out_dim,
    W_std=np.sqrt(2),
    b_std=0.1,
    parameterization="ntk",
):
    """Returns neural_tangents.stax fully connected network."""
    activation_fn = stax.Relu()
    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization
    )

    layers = []
    for hidden in hidden_dims:
        layers += [dense(hidden), activation_fn]
    layers += [
        stax.Dense(out_dim, W_std=W_std, b_std=b_std, parameterization=parameterization)
    ]

    return stax.serial(*layers)


def get_kernel_fn(hidden_dims, out_dim):
    _, _, _kernel_fn = FullyConnectedNetwork(hidden_dims, out_dim)
    kernel_fn = jax.jit(functools.partial(_kernel_fn, get="ntk"))
    return kernel_fn


def get_loss_fn(kernel_fn):
    @jax.jit
    def loss_fn(x_support, y_support, x_target, y_target, reg=1e-6):
        k_ss = kernel_fn(x_support, x_support)
        k_ts = kernel_fn(x_target, x_support)
        k_ss_reg = (
            k_ss
            + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0]
        )
        pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, assume_a="pos"))
        mse_loss = 0.5 * jnp.mean((pred - y_target) ** 2)
        acc = jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(y_target, axis=1))
        return mse_loss, acc

    return loss_fn


def get_update_functions(init_params, kernel_fn, lr):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(init_params)
    loss_fn = get_loss_fn(kernel_fn)
    value_and_grad = jax.value_and_grad(
        lambda params, x_target, y_target: loss_fn(
            params["x"], params["y"], x_target, y_target
        ),
        has_aux=True,
    )

    @jax.jit
    def update_fn(step, opt_state, params, x_target, y_target):
        (loss, acc), dparams = value_and_grad(params, x_target, y_target)
        return opt_update(step, dparams, opt_state), (loss, acc)

    return opt_state, get_params, update_fn


def convert_onehot(labels):
    converted = np.zeros((len(labels), len(set(labels))))
    converted[np.arange(len(labels)), labels] = 1
    return converted


def kip(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    n_epochs: int,
    mlp_dim: int,
    random_state=0,
) -> tuple[np.ndarray, np.ndarray]:
    idxs = np.arange(len(X))
    X = X.astype(float)
    y_onehot = convert_onehot(y).astype(float)

    support_idxs, _ = random_sample(idxs, y, N, random_state=random_state)
    support_idxs = support_idxs.flatten()
    x_support = X[support_idxs].copy()
    y_support = y_onehot[support_idxs].copy()

    params_init = {"x": x_support, "y": y_support}

    kernel_fn = get_kernel_fn([mlp_dim], y_onehot.shape[1])
    (opt_state, get_params, update_fn) = get_update_functions(
        params_init, kernel_fn, 4e-2
    )
    params = get_params(opt_state)

    for i in range(n_epochs):
        target_idxs, _ = random_sample(
            idxs,
            y,
            N * 10,
            random_state=random_state,
        )
        target_idxs = target_idxs.flatten()
        x_target = X[target_idxs]
        y_target = y_onehot[target_idxs]
        opt_state, (train_loss, train_acc) = update_fn(
            i + 1, opt_state, params, x_target, y_target
        )
        params = get_params(opt_state)

    return np.array(params["x"]), params_init["y"].argmax(1)
