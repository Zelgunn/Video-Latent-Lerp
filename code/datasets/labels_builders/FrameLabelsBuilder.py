from typing import Union, List
import numpy as np

from datasets.data_readers import VideoReader


class FrameLabelsBuilder(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str]],
                 shard_duration: float,
                 frequency: float
                 ):

        if isinstance(labels_source, str):
            if labels_source.endswith(".npy") or labels_source.endswith(".npz"):
                labels_source = np.load(labels_source, mmap_mode="r")
            else:
                labels_source = VideoReader(labels_source)

        elif isinstance(labels_source, list):
            labels_source = VideoReader(labels_source)

        self.shard_duration = shard_duration
        self.state_duration = 1.0 / frequency
        self.labels_source = labels_source

    def __iter__(self):
        previous_state = False
        labels = []
        start = None

        time = 0

        for frame in self.labels_source:
            state = np.any(frame.astype(np.float) > 0.5)

            if state != previous_state:
                position = (time / self.shard_duration) if time > 0.0 else 0.0
                if state:
                    start = position
                else:
                    labels += [start, position]

            previous_state = state
            time += self.state_duration

            if (time + self.state_duration) > self.shard_duration:
                if state:
                    labels += [start, 1.0]
                time -= self.shard_duration

                yield labels

                labels = []
                previous_state = False

        if time > 0:
            if previous_state:
                labels += [start, 1.0]
            yield labels
