import tensorflow as tf
from tensorflow.python.keras import Model
import os
from time import time
from typing import Dict

from modalities import RawVideo


def get_log_dir(base_dir):
    log_dir = os.path.join(base_dir, "log_{0}".format(int(time())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_model_json(model: Model, log_dir):
    filename = "{0}_config.json".format(model.name)
    filename = os.path.join(log_dir, filename)
    with open(filename, "w") as file:
        file.write(model.to_json())


def save_model_summary(model: Model, log_dir):
    filename = "{0}_summary.txt".format(model.name)
    filename = os.path.join(log_dir, filename)
    with open(filename, "w") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))


def save_model_info(model: Model, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # keras.utils.plot_model(model, os.path.join(log_dir, "{0}.png".format(model.name)))
    save_model_json(model, log_dir)
    save_model_summary(model, log_dir)


def augment_raw_video(modalities: Dict[str, tf.Tensor]
                      ) -> Dict[str, tf.Tensor]:
    if RawVideo.id() not in modalities:
        return modalities

    raw_video = modalities[RawVideo.id()]

    # raw_video = random_video_vertical_flip(raw_video)
    raw_video = random_video_horizontal_flip(raw_video)
    raw_video = tf.image.random_hue(raw_video, max_delta=0.1)
    raw_video = tf.image.random_brightness(raw_video, max_delta=0.1)

    modalities[RawVideo.id()] = raw_video
    return modalities


# region Random video flip
def random_video_vertical_flip(video: tf.Tensor,
                               seed: int = None,
                               scope_name: str = "random_video_vertical_flip"
                               ) -> tf.Tensor:
    return random_video_flip(video, 1, seed, scope_name)


def random_video_horizontal_flip(video: tf.Tensor,
                                 seed: int = None,
                                 scope_name: str = "random_video_horizontal_flip"
                                 ) -> tf.Tensor:
    return random_video_flip(video, 2, seed, scope_name)


def random_video_flip(video: tf.Tensor,
                      flip_index: int,
                      seed: int,
                      scope_name: str
                      ) -> tf.Tensor:
    """Randomly (50% chance) flip an video along axis `flip_index`.
    Args:
        video: 5-D Tensor of shape `[batch, time, height, width, channels]` or
               4-D Tensor of shape `[time, height, width, channels]`.
        flip_index: Dimension along which to flip video. Time: 0, Vertical: 1, Horizontal: 2
        seed: A Python integer. Used to create a random seed. See `tf.set_random_seed` for behavior.
        scope_name: Name of the scope in which the ops are added.
    Returns:
        A tensor of the same type and shape as `video`.
    Raises:
        ValueError: if the shape of `video` not supported.
    """
    with tf.name_scope(scope_name) as scope:
        video = tf.convert_to_tensor(video, name="video")
        shape = video.get_shape()

        if shape.ndims == 4:
            uniform_random = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, seed=seed)
            flip_condition = tf.less(uniform_random, 0.5)
            flipped = tf.reverse(video, [flip_index])
            outputs = tf.cond(pred=flip_condition,
                              true_fn=lambda: flipped,
                              false_fn=lambda: video,
                              name=scope)

        elif shape.ndims == 5:
            batch_size = tf.shape(video)[0]
            uniform_random = tf.random.uniform(shape=[batch_size], minval=0.0, maxval=1.0, seed=seed)
            uniform_random = tf.reshape(uniform_random, [batch_size, 1, 1, 1, 1])
            flips = tf.round(uniform_random)
            flips = tf.cast(flips, video.dtype)
            flipped = tf.reverse(video, [flip_index + 1])
            outputs = flips * flipped + (1.0 - flips) * video

        else:
            raise ValueError("`video` must have either 4 or 5 dimensions but has {} dimensions.".format(shape.ndims))

        return outputs


# endregion

def add_gaussian_noise(modalities: Dict[str, tf.Tensor],
                       noise_mean=0.0,
                       noise_stddev=0.1,
                       min_val=0.0,
                       max_val=1.0
                       ) -> Dict[str, tf.Tensor]:
    if RawVideo.id() not in modalities:
        return modalities

    raw_video = modalities[RawVideo.id()]

    noise = tf.random.normal(tf.shape(raw_video), mean=noise_mean, stddev=noise_stddev, name="gaussian_noise")
    if min_val is not None:
        raw_video = tf.clip_by_value(raw_video + noise, min_val, max_val)

    modalities[RawVideo.id()] = raw_video
    return modalities
