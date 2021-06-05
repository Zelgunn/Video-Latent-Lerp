import tensorflow as tf


@tf.function
def diff(tensor, axis=-1):
    if axis < 0:
        axis = tf.rank(tensor) + axis

    offset = tf.zeros([axis], dtype=tf.int32)
    offset = tf.concat([offset, [1]], axis=0)

    partial_shape = tf.shape(tensor)[:axis + 1]

    left = tf.strided_slice(tensor, begin=tf.zeros_like(offset), end=partial_shape - offset)
    right = tf.strided_slice(tensor, begin=offset, end=partial_shape)

    return right - left
