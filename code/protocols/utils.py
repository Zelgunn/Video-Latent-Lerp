import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LeakyReLU, Layer, Flatten, Dense
from tensorflow.python.keras.layers.convolutional import Conv
from typing import List, Tuple, Union

from CustomKerasLayers import ResBlockND, ResBlockNDTranspose


# region Autoencoder
def make_residual_encoder(input_shape: Tuple[int, int, int, int],
                          filters: List[int],
                          kernel_size: int,
                          strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                          code_size: int,
                          code_activation: Union[str, Layer],
                          use_batch_norm: bool,
                          use_residual_bias: bool,
                          use_conv_bias: bool,
                          name="ResidualEncoder") -> Sequential:
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    leaky_relu = LeakyReLU(alpha=1e-2)
    kwargs = {
        "input_shape": input_shape,
        "activation": leaky_relu,
        "use_batch_norm": use_batch_norm,
        "use_residual_bias": use_residual_bias,
        "use_conv_bias": use_conv_bias,
        "rank": rank,
    }
    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        layer = ResBlockND(filters=filters[i], strides=strides[i], kernel_size=kernel_size, **kwargs)
        layers.append(layer)

        if i == 0:
            kwargs.pop("input_shape")

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

    kwargs["activation"] = code_activation
    layers.append(ResBlockND(filters=code_size, strides=1, kernel_size=kernel_size, **kwargs))

    return Sequential(layers=layers, name=name)


def make_residual_decoder(input_shape: Tuple[int, int, int, int],
                          filters: List[int],
                          kernel_size: int,
                          strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                          channels: int,
                          output_activation: Union[str, Layer],
                          use_batch_norm: bool,
                          use_residual_bias: bool,
                          use_conv_bias: bool,
                          name="ResidualDecoder") -> Sequential:
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    leaky_relu = LeakyReLU(alpha=1e-2)
    kwargs = {
        "input_shape": input_shape,
        "activation": leaky_relu,
        "use_residual_bias": use_residual_bias,
        "use_conv_bias": use_conv_bias,
        "use_batch_norm": use_batch_norm,
        "rank": rank,
    }
    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        layer = ResBlockNDTranspose(filters=filters[i], strides=strides[i], kernel_size=kernel_size, **kwargs)
        layers.append(layer)

        if i == 0:
            kwargs.pop("input_shape")

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

    kwargs.pop("use_conv_bias")
    kwargs.pop("use_residual_bias")
    kwargs.pop("use_batch_norm")
    kwargs["kernel_size"] = 1
    kwargs["activation"] = output_activation
    layers.append(Conv(filters=channels, strides=1, padding="same", use_bias=False, **kwargs))

    return Sequential(layers=layers, name=name)


def make_discriminator(input_shape: Tuple[int, int, int, int],
                       filters: List[int],
                       kernel_size: int,
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       intermediate_size: int,
                       intermediate_activation: str,
                       include_intermediate_output: bool,
                       name="Discriminator"):
    # region Core : [Conv(), Conv(), ..., Flatten(), Dense()]
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    leaky_relu = LeakyReLU(alpha=1e-2)
    kwargs = {
        "input_shape": input_shape,
        "activation": leaky_relu,
        "rank": rank,
        "padding": "valid",
    }
    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        layer = Conv(filters=filters[i], strides=strides[i], kernel_size=kernel_size, **kwargs)
        layers.append(layer)

        if i == 0:
            kwargs.pop("input_shape")

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

    layers.append(Flatten())
    layers.append(Dense(units=intermediate_size, activation=intermediate_activation))

    core_model = Sequential(layers, name="{}Core".format(name))
    # endregion

    input_layer = core_model.input
    intermediate_output = core_model.output
    final_output = Dense(units=1, activation="linear")(intermediate_output)

    outputs = [final_output, intermediate_output] if include_intermediate_output else [final_output]

    return Model(inputs=[input_layer], outputs=outputs, name=name)


def validate_filters_and_strides(filters: List[int],
                                 strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]]
                                 ):
    if len(filters) != len(strides):
        raise ValueError("`filters` and `strides` must have the same length. "
                         "Found {} and {}".format(len(filters), len(strides)))


# endregion

# region Data augmentation

def video_random_cropping(video: tf.Tensor, crop_ratio: float, output_length: int):
    crop_ratio = tf.random.uniform(shape=(), minval=1.0 - crop_ratio, maxval=1.0)
    original_shape = tf.cast(tf.shape(video), tf.float32)
    original_height, original_width = original_shape[1], original_shape[2]

    crop_size = [output_length, crop_ratio * original_height, crop_ratio * original_width, 1]

    video = tf.image.random_crop(video, crop_size)
    return video


def dropout_noise(inputs, max_rate, noise_shape_prob):
    """
    :param inputs: A floating point `Tensor`.
    :param max_rate: A floating point scalar `Tensor`. The maximum probability that each element is dropped.
    :param noise_shape_prob: A 1-D `Tensor` of type `float32`, representing the probability of dropping each
        dimension completely.
    :return: A `Tensor` of the same shape and type as `inputs.
    """
    noise_shape = tf.random.uniform(shape=[len(inputs.shape)], minval=0.0, maxval=1.0, dtype=tf.float32)
    noise_shape = noise_shape < noise_shape_prob
    noise_shape = tf.cast(noise_shape, tf.int32)
    noise_shape = tf.shape(inputs) * (1 - noise_shape) + noise_shape

    rate = tf.random.uniform(shape=[], minval=0.0, maxval=max_rate, dtype=tf.float32)
    random_tensor = tf.random.uniform(shape=noise_shape, minval=0.0, maxval=1.0, dtype=tf.float32)
    keep_mask = random_tensor >= rate

    outputs = inputs * tf.cast(keep_mask, inputs.dtype) / (1.0 - rate)
    return outputs


def video_dropout_noise(video, max_rate, spatial_prob):
    drop_width = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5
    drop_width = tf.cast(drop_width, tf.float32)

    noise_shape_prob = [spatial_prob * (1.0 - drop_width), spatial_prob * drop_width, 1.0]
    noise_shape_prob = [0.0] * (len(video.shape) - 3) + noise_shape_prob

    video = dropout_noise(video, max_rate, noise_shape_prob)
    return video


def random_image_negative(image, negative_prob=0.5):
    negative = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32) < negative_prob
    image = tf.cond(pred=negative,
                    true_fn=lambda: 1.0 - image,
                    false_fn=lambda: image)
    return image

# endregion
