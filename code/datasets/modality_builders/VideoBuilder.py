import numpy as np
import cv2
from typing import Union, List, Tuple, Any, Dict, Type, Optional

from modalities import Modality, ModalityCollection, RawVideo, OpticalFlow, DoG
from datasets.modality_builders import ModalityBuilder
from datasets.data_readers.VideoReader import VideoReader, VideoReaderProto


class VideoBuilder(ModalityBuilder):
    def __init__(self,
                 shard_duration: float,
                 source_frequency: Union[int, float],
                 modalities: ModalityCollection,
                 video_reader: Union[VideoReader, Any],
                 default_frame_size: Union[Tuple[int, int], List[int], None],
                 buffer_frame_size: Union[Tuple[int, int], List[int], None],
                 ):

        super(VideoBuilder, self).__init__(shard_duration=shard_duration,
                                           source_frequency=source_frequency,
                                           modalities=modalities)

        if isinstance(video_reader, VideoReaderProto):
            video_reader = video_reader.to_video_reader()
        elif not isinstance(video_reader, VideoReader):
            video_reader = VideoReader(video_reader)
        else:
            video_reader = video_reader

        self.reader = video_reader
        self.default_frame_size = default_frame_size
        self.buffer_frame_size = buffer_frame_size

        self.frame_count = video_reader.frame_count
        if OpticalFlow in self.modalities or DoG in self.modalities:
            self.frame_count -= 1
            self.skip_first = True
        else:
            self.skip_first = False

    @classmethod
    def supported_modalities(cls):
        return [RawVideo, OpticalFlow, DoG]

    # region Frame/Shard processing
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.buffer_frame_size is not None:
            frame = self.resize_frame(frame, self.buffer_frame_size)
        return frame

    def check_shard(self, frames: np.ndarray) -> bool:
        return frames.shape[0] > 1

    def resize_frame(self,
                     frame: np.ndarray,
                     frame_size: Tuple[int, int] = None
                     ) -> np.ndarray:
        if frame_size is None:
            frame_size = self.default_frame_size

        if frame_size is not None:
            if tuple(frame_size) != frame.shape[:2]:
                frame = cv2.resize(frame, dsize=tuple(reversed(frame_size)))
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)
        return frame

    def resize_frames(self,
                      frames: np.ndarray,
                      frame_size: Tuple[int, int] = None
                      ) -> np.ndarray:
        if frame_size is None:
            frame_size = self.default_frame_size

        if frame_size is None:
            return frames

        frame_count = frames.shape[0]
        channels = frames.shape[-1] if frames.ndim > 3 else 1
        outputs = np.empty(shape=(frame_count, *frame_size, channels), dtype=np.float32)

        for i in range(frame_count):
            outputs[i] = self.resize_frame(frames[i])

        return outputs

    def process_shard(self, frames: np.ndarray) -> Dict[Type[Modality], np.ndarray]:
        shard: Dict[Type[Modality], np.ndarray] = {}
        frames = frames.astype(np.float64)

        if RawVideo in self.modalities:
            shard[RawVideo] = self.resize_frames(frames)

        if OpticalFlow in self.modalities:
            optical_flow: OpticalFlow = self.modalities[OpticalFlow]
            shard[OpticalFlow] = optical_flow.compute_flow(frames, self.default_frame_size)

        if DoG in self.modalities:
            dog: DoG = self.modalities[DoG]
            shard[DoG] = dog.compute_difference_of_gaussians(frames, self.default_frame_size)

        return shard

    # endregion

    def get_buffer_shape(self, frame: np.ndarray = None) -> Optional[List[int]]:
        if self.buffer_frame_size is not None:
            frame_size = self.buffer_frame_size
        elif frame is not None:
            frame_size = frame.shape[:2]
        else:
            frame_size = None

        if frame_size is None:
            return None

        max_shard_size = self.get_source_max_shard_size()
        return [max_shard_size, *frame_size, self.reader.frame_channels]

    def get_frame_size(self, none_if_reader_default=False):
        if self.default_frame_size is not None:
            return self.default_frame_size
        else:
            if none_if_reader_default:
                return None
            else:
                return self.reader.frame_size

    @property
    def source_frame_count(self):
        return self.reader.frame_count
