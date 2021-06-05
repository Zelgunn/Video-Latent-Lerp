import tensorflow as tf
import cv2
import numpy as np
from typing import Dict, Tuple

from modalities import Modality


class RawVideo(Modality):
    def __init__(self):
        super(RawVideo, self).__init__()

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        if modality_value.ndim == 3:
            modality_value = np.expand_dims(modality_value, axis=-1)

        return {cls.id(): video_feature(modality_value)}

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features: Dict[str, tf.SparseTensor]):
        encoded_raw_video: tf.Tensor = parsed_features[cls.id()].values

        raw_video_shard_size = tf.shape(encoded_raw_video)[0]
        raw_video = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(encoded_raw_video[i]), tf.float32),
                              tf.range(raw_video_shard_size),
                              dtype=tf.float32)

        return raw_video

    @classmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        return {cls.id(): tf.io.VarLenFeature(tf.string)}

    @classmethod
    def rank(cls) -> int:
        return 4

    @staticmethod
    def compute_raw_frame(frame: np.ndarray,
                          frame_size: Tuple[int, int]):
        raw = frame

        if frame_size is not None:
            if tuple(frame_size) != frame.shape[:2]:
                raw = cv2.resize(raw, dsize=tuple(reversed(frame_size)))

        if raw.ndim == 2:
            raw = np.expand_dims(raw, axis=-1)

        return raw


def video_feature(video):
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes()) for frame in video]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_frames))
