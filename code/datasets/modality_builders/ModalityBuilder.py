import numpy as np
from abc import ABC, abstractmethod
from typing import Type, Dict, Union, Iterable, Optional

from modalities import Modality, ModalityCollection, RawVideo, RawAudio, DoG, OpticalFlow
from misc_utils.misc_utils import int_ceil, int_floor

EPSILON = 1e-5


class ModalityBuilder(ABC):
    def __init__(self,
                 shard_duration: float,
                 source_frequency: Union[int, float],
                 modalities: ModalityCollection):
        modalities = self.filter_supported_modalities(modalities)

        self.shard_duration = shard_duration
        self.source_frequency = source_frequency

        self.modalities = modalities
        self.reader: Optional[Iterable] = None

    # region Supported modalities
    @classmethod
    @abstractmethod
    def supported_modalities(cls):
        raise NotImplementedError

    @classmethod
    def supports(cls, modality: Modality) -> bool:
        if not isinstance(modality, Modality):
            raise ValueError("`modality` is not a Modality, got type {}.".format(type(modality)))
        return type(modality) in cls.supported_modalities()

    @classmethod
    def supports_any(cls, modalities: ModalityCollection) -> bool:
        return any([cls.supports(modality) for modality in modalities])

    @classmethod
    def filter_supported_modalities(cls, modalities: ModalityCollection) -> ModalityCollection:
        filtered_modalities = []
        for modality in modalities:
            if cls.supports(modality):
                filtered_modalities.append(modality)
        return ModalityCollection(filtered_modalities)

    # endregion

    def __iter__(self):
        shard_buffer = self.get_shard_buffer()
        source_shard_size = self.get_source_initial_shard_size()

        i = 0
        time = 0.0

        if self.reader is None:
            raise ValueError("You must provide a reader for the builder")

        for frame in self.reader:
            if shard_buffer is None:
                shard_buffer = self.get_shard_buffer(frame)

            shard_buffer[i] = self.process_frame(frame)

            i += 1
            if (i % source_shard_size) == 0:
                frames = shard_buffer[:source_shard_size]
                if self.check_shard(frames):
                    shard = self.process_shard(frames)
                    yield shard

                time += self.shard_duration
                source_shard_size = self.get_source_next_shard_size(time)
                i = 0

    # region Frame/Shard processing
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    @abstractmethod
    def check_shard(self, frames: np.ndarray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process_shard(self, frames: np.ndarray) -> Dict[Type[Modality], np.ndarray]:
        raise NotImplementedError

    # endregion

    # region Buffer
    def get_shard_buffer(self, frame: np.ndarray = None) -> Optional[np.ndarray]:
        buffer_shape = self.get_buffer_shape(frame)
        if buffer_shape is None:
            return None
        return np.zeros(buffer_shape, dtype="float32")

    @abstractmethod
    def get_buffer_shape(self, frame: np.ndarray = None):
        raise NotImplementedError

    # endregion

    # region Shard size (initial, max, next)
    def compute_shard_size(self, modality: Modality, source_shard_size: int) -> int:
        if isinstance(modality, RawVideo) or isinstance(modality, RawAudio):
            return source_shard_size
        elif isinstance(modality, OpticalFlow) or isinstance(modality, DoG):
            return source_shard_size - 1
        else:
            raise NotImplementedError(modality.id())

    def get_source_max_shard_size(self) -> int:
        return get_max_frame_count(self.shard_duration, self.source_frequency)

    def get_source_initial_shard_size(self) -> int:
        return get_min_frame_count(self.shard_duration, self.source_frequency)

    def get_source_next_shard_size(self, time: float):
        yielded_frame_count = get_min_frame_count(time, self.source_frequency)
        total_frame_count_yielded_next_time = get_min_frame_count(time + self.shard_duration, self.source_frequency)

        if total_frame_count_yielded_next_time > self.source_frame_count:
            total_frame_count_yielded_next_time = self.source_frame_count

        shard_size = total_frame_count_yielded_next_time - yielded_frame_count

        return shard_size

    @property
    @abstractmethod
    def source_frame_count(self):
        raise NotImplementedError

    # endregion


def get_max_frame_count(duration: float, frequency: Union[int, float]):
    return int_ceil(duration * frequency, EPSILON)


def get_min_frame_count(duration: float, frequency: Union[int, float]):
    return int_floor(duration * frequency, EPSILON)
