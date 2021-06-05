import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, NamedTuple, Type

from modalities import int64_list_feature, bytes_list_feature


class Modality(ABC):
    def __init__(self, **kwargs):
        pass

    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)

    @classmethod
    @abstractmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        raise NotImplementedError

    @classmethod
    def tfrecord_shape_parse_function(cls):
        return tf.io.VarLenFeature(dtype=tf.int64)

    @classmethod
    def encode_raw(cls, array: np.ndarray, dtype: Union[type, str]) -> Dict[str, tf.train.Feature]:
        features = {
            cls.shape_id(): int64_list_feature(array.shape),
            cls.id(): bytes_list_feature([array.astype(dtype).tobytes()])
        }
        return features

    @classmethod
    def decode_raw(cls, parsed_features: Dict[str, tf.SparseTensor], dtype: tf.dtypes.DType):
        encoded_modality = parsed_features[cls.id()].values
        modality_shape = parsed_features[cls.shape_id()].values

        def shape_non_empty():
            return tf.size(modality_shape) > 0

        def decode_and_reshape():
            decoded = tf.io.decode_raw(encoded_modality, dtype)
            return tf.reshape(decoded, modality_shape, name="reshape_decoded_{}".format(cls.id()))

        def empty():
            return tf.constant([], dtype=dtype, name="empty_{}".format(cls.id()))

        decoded_modality = tf.cond(pred=shape_non_empty(),
                                   true_fn=decode_and_reshape,
                                   false_fn=empty)

        return decoded_modality

    @classmethod
    def id(cls):
        return cls.__name__

    @classmethod
    def shape_id(cls):
        return cls.id() + "_shape"

    @classmethod
    @abstractmethod
    def rank(cls) -> int:
        raise NotImplementedError


class ModalityLoadInfo(NamedTuple):
    modality: Type[Modality]
    length: int
    # output_shape: Tuple[int, ...]
    # output_map: Callable[[tf.Tensor, Dict[str, tf.Tensor]], tf.Tensor] = None

    # @property
    # def rank(self) -> int:
    #     return len(self.output_shape)
