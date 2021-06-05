import tensorflow as tf
import numpy as np


def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def int_ceil(value, epsilon=1e-5) -> int:

    return int(np.ceil(value - epsilon))


def int_floor(value, epsilon=1e-5) -> int:

    return int(np.floor(value + epsilon))


def get_known_shape(tensor: tf.Tensor):
    dyn_shape = tf.shape(tensor)
    outputs_shape = [dyn_shape[i] if tensor.shape[i] is None else tensor.shape[i]
                     for i in range(len(tensor.shape))]
    return outputs_shape
