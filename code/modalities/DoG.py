import tensorflow as tf
import numpy as np
import cv2
from typing import Dict, Any, Tuple

from modalities import Modality


class DoG(Modality):
    def __init__(self, blurs=(2.0, 2.82, 4.0, 5.66, 8.0)):
        super(DoG, self).__init__()
        self.blurs = blurs

    def get_config(self) -> Dict[str, Any]:
        base_config = super(DoG, self).get_config()
        config = {"blurs": list(self.blurs)}
        return {**base_config, **config}

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        return cls.encode_raw(modality_value, np.float16)

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        return cls.decode_raw(parsed_features, tf.float16)

    @classmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        return {cls.id(): tf.io.VarLenFeature(tf.string),
                cls.shape_id(): cls.tfrecord_shape_parse_function()}

    @classmethod
    def rank(cls) -> int:
        return 4

    def compute_difference_of_gaussians(self,
                                        frames: np.ndarray,
                                        frame_size: Tuple[int, int]
                                        ) -> np.ndarray:
        frame_count = frames.shape[0]
        buffer = np.empty(shape=[frame_count, *frame_size, len(self.blurs) - 1])

        for i in range(frame_count):
            buffer[i] = self.compute_difference_of_gaussians_frame(frame=frames[i],
                                                                   frame_size=frame_size)
        return buffer

    def compute_difference_of_gaussians_frame(self,
                                              frame: np.ndarray,
                                              frame_size: Tuple[int, int]
                                              ) -> np.ndarray:
        if frame.ndim == 3:
            if frame.shape[2] == 1:
                frame = np.squeeze(frame, axis=2)
            else:
                frame = np.mean(frame, axis=2)

        frames = [None] * len(self.blurs)
        for i in range(len(self.blurs)):
            frames[i] = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=self.blurs[i])
            if frame_size is not None:
                frames[i] = cv2.resize(frames[i], dsize=tuple(reversed(frame_size)))

        deltas = [None] * (len(self.blurs) - 1)

        for i in range(len(deltas)):
            deltas[i] = np.abs(frames[i] - frames[i + 1])

        deltas = np.stack(deltas, axis=-1)
        return deltas
