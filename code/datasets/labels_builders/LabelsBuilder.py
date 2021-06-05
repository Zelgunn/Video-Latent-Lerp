from typing import Union, Tuple, List, Optional
import numpy as np
from enum import IntEnum

from datasets.labels_builders import SingleValueLabelsBuilder, TimestampsLabelsBuilder, FrameLabelsBuilder


class LabelsBuilderMode(IntEnum):
    SINGLE_VALUE = 0,
    TIMESTAMPS = 1,
    FRAMES = 2


class LabelsBuilder(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, bool, int, float, List[Tuple[float, float]]],
                 shard_count: int,
                 shard_duration: float,
                 frequency: float,
                 mode: LabelsBuilderMode = None
                 ):

        if mode is None:
            mode = LabelsBuilder.define_mode_from_source(labels_source)

        self.mode = mode

        self.shard_count = shard_count
        self.shard_duration = shard_duration
        self.frequency = frequency

        self.single_value_labels_builder: Optional[SingleValueLabelsBuilder] = None
        self.timestamps_labels_builder: Optional[TimestampsLabelsBuilder] = None
        self.frame_labels_builder: Optional[FrameLabelsBuilder] = None

        if self.mode == LabelsBuilderMode.SINGLE_VALUE:
            self.single_value_labels_builder = SingleValueLabelsBuilder(labels_source, shard_count=shard_count)
        elif self.mode == LabelsBuilderMode.TIMESTAMPS:
            self.timestamps_labels_builder = TimestampsLabelsBuilder(labels_source, shard_duration=shard_duration,
                                                                     shard_count=shard_count)
        elif self.mode == LabelsBuilderMode.FRAMES:
            if frequency is None:
                raise ValueError("You must provide a value for `frequency` when using frame labels.")
            self.frame_labels_builder = FrameLabelsBuilder(labels_source, shard_duration=shard_duration,
                                                           frequency=frequency)

    @staticmethod
    def define_mode_from_source(labels_source):
        mode = None
        if isinstance(labels_source, str):
            if labels_source.endswith(".npy") or labels_source.endswith(".npz"):
                labels_source = np.load(labels_source, mmap_mode="r")
            else:
                mode = LabelsBuilderMode.FRAMES

        elif isinstance(labels_source, int) or \
                isinstance(labels_source, float) or \
                isinstance(labels_source, bool):
            mode = LabelsBuilderMode.SINGLE_VALUE

        elif isinstance(labels_source, list):
            labels_source = np.asarray(labels_source)

        if isinstance(labels_source, np.ndarray):
            if labels_source.ndim == 0:
                mode = LabelsBuilderMode.SINGLE_VALUE
            elif labels_source.ndim == 1:
                mode = LabelsBuilderMode.FRAMES
            elif labels_source.ndim == 2:
                if labels_source.shape[1] == 2:
                    mode = LabelsBuilderMode.TIMESTAMPS
                else:
                    mode = LabelsBuilderMode.FRAMES
            else:
                mode = LabelsBuilderMode.FRAMES

        if mode is None:
            raise ValueError("mode could not be determined from source of type {}".format(type(labels_source)))

        return mode

    def __iter__(self):
        if self.mode == LabelsBuilderMode.SINGLE_VALUE:
            return iter(self.single_value_labels_builder)
        elif self.mode == LabelsBuilderMode.TIMESTAMPS:
            return iter(self.timestamps_labels_builder)
        elif self.mode == LabelsBuilderMode.FRAMES:
            return iter(self.frame_labels_builder)
