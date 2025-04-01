from typing import OrderedDict

import numpy as np

import archimedes as arc
from archimedes._core import FunctionCache

__all__ = [
    "rbf",
    "relu",
    "elu",
    "sigmoid",
    "swish",
    "adam",
    "dense",
]


def rbf(x):
    return np.exp(-(x**2))


def relu(x):
    return np.maximum(x, 0)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def swish(x):
    return x * sigmoid(x)


def adam(
    obj,
    x0,
    iters,
    args=(),
    static_argnames=None,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    print_every=None,
    callback=None,
    callback_iters=None,
):
    if not isinstance(obj, FunctionCache):
        obj = FunctionCache(obj, static_argnames=static_argnames)

    grad = arc.grad(obj)
    x = x0.copy()
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)

    if callback_iters is not None and callback is None:

        def callback(i, x, J):
            return J

    history = OrderedDict({})
    for i in range(iters):
        g = grad(x, *args)
        # x = x - lr * g  # Basic gradient descent
        if weight_decay != 0:
            g = g + weight_decay * x
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g**2
        mhat = m / (1 - betas[0] ** (i + 1))
        vhat = v / (1 - betas[1] ** (i + 1))
        x = x - lr * mhat / (vhat**0.5 + eps)

        if print_every is not None and (i + 1) % print_every == 0:
            print(f"Iter {i + 1}/{iters}: {obj(x, *args)}")

        if callback_iters is not None and (i + 1) % callback_iters == 0:
            J = obj(x, *args)
            history[i + 1] = callback(i + 1, x, J)

    return x, history


def glorot_normal(rng, n_in, n_out):
    scale = np.sqrt(2 / (n_in + n_out))
    return scale * rng.normal(size=(n_out, n_in))


def zero_initialization(rng, n_in, n_out):
    return np.zeros((n_out, n_in))


INITIALIZATIONS = {
    "glorot_normal": glorot_normal,
    "zero": zero_initialization,
}


def dense(
    num_inputs,
    width,
    depth,
    num_outputs,
    activation=np.tanh,
    activate_final=False,
    rng=None,
    kind="MX",
    initialization="glorot_normal",
):
    if rng is None:
        rng = np.random.default_rng()

    if initialization not in INITIALIZATIONS:
        raise ValueError(f"Unknown initialization {initialization}")
    init_weights = INITIALIZATIONS[initialization]

    weights = [init_weights(rng, num_inputs, width)]
    biases = [np.zeros(width)]

    for _ in range(depth - 1):
        weights.append(init_weights(rng, width, width))
        biases.append(np.zeros(width))

    weights.append(init_weights(rng, width, num_outputs))
    biases.append(np.zeros(num_outputs))

    p = np.concatenate([W.ravel() for W in weights] + biases)

    if activation is None:

        def activation(x):
            return x

    def forward(x, p):
        W_ = []
        b_ = []
        idx = 0

        for w in weights:
            W_.append(np.reshape(p[idx : idx + w.size], w.shape))
            idx += w.size
        for b in biases:
            b_.append(p[idx : idx + b.size, None])
            idx += b.size

        # Allowable input shapes are (n_in,) or (n_in, n_samples) for vectorization.
        # Cannot call with scalar inputs, even if num_inputs==1
        # Support vectorization by calling with (n_in, n_samples) arrays
        if x.ndim == 0:
            raise ValueError("Cannot evaluate dense network with scalar input.")

        # Either (n_in,) or (n_in, n_samples,)
        if x.shape[0] != num_inputs:
            raise ValueError(
                f"Input shape {x.shape} incompatible with num_inputs={num_inputs}"
            )

        if x.ndim == 1:
            num_samples = 1
            flatten_output = True
        else:
            num_samples = x.shape[1]
            flatten_output = False

        x = np.reshape(x, (num_inputs, num_samples))

        # Forward pass
        for i, (w, b) in enumerate(zip(W_, b_)):
            x = np.dot(w, x) + b
            if i < depth or activate_final:
                x = activation(x)

        # x has shape (num_outputs, num_samples)
        # If the input was 1D, flatten the output
        if flatten_output:
            x = np.reshape(x, (num_outputs,))

        return x

    forward = FunctionCache(
        forward,
        arg_names=("x", "p"),
        kind=kind,
    )

    return p, forward
