import tensorflow as tf
import numpy as np
import cv2
from typing import Dict, Any, Union, Tuple

from modalities import Modality


class OpticalFlow(Modality):
    def __init__(self,
                 use_polar: bool,
                 pyr_scale=0.5,
                 levels=3,
                 winsize=5,
                 iterations=5,
                 poly_n=5,
                 poly_sigma=1.2):
        super(OpticalFlow, self).__init__()
        self.use_polar = use_polar
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def get_config(self) -> Dict[str, Any]:
        base_config = super(OpticalFlow, self).get_config()
        config = \
            {
                "use_polar": self.use_polar,
                **self.farneback_params
            }
        return {**base_config, **config}

    @property
    def farneback_params(self) -> Dict[str, Union[float, int]]:
        return {
            "pyr_scale": self.pyr_scale,
            "levels": self.levels,
            "winsize": self.winsize,
            "iterations": self.iterations,
            "poly_n": self.poly_n,
            "poly_sigma": self.poly_sigma
        }

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

    # TODO : Compute flow with other tools (copy from preprocessed file, from model, ...)
    def compute_flow(self,
                     frames: np.ndarray,
                     frame_size: Tuple[int, int]
                     ) -> np.ndarray:
        frame_count = frames.shape[0] - 1
        buffer = np.empty(shape=[frame_count, *frame_size, 2])

        for i in range(frame_count):
            buffer[i] = self.compute_flow_frame(previous_frame=frames[i],
                                                frame=frames[i + 1],
                                                frame_size=frame_size)
        return buffer

    def compute_flow_frame(self,
                           previous_frame: np.ndarray,
                           frame: np.ndarray,
                           frame_size: Tuple[int, int]
                           ) -> np.ndarray:
        if frame.ndim == 3:
            frame = frame.mean(axis=-1)
            previous_frame = previous_frame.mean(axis=-1)

        flow: np.ndarray = cv2.calcOpticalFlowFarneback(prev=previous_frame, next=frame, flow=None,
                                                        flags=0, **self.farneback_params)

        absolute_flow = np.abs(flow)
        if np.min(absolute_flow) < 1e-20:
            flow[absolute_flow < 1e-20] = 0.0

        if self.use_polar:
            flow = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
            flow = np.stack(flow, axis=-1)

        if frame_size is not None:
            flow = cv2.resize(flow, dsize=tuple(reversed(frame_size)))

        return flow
