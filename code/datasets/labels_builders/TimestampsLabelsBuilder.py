from typing import Union, List, Tuple
import numpy as np


class TimestampsLabelsBuilder(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[Tuple[float, float]]],
                 shard_duration: float,
                 shard_count: int
                 ):

        if isinstance(labels_source, str):
            labels_source = np.load(labels_source, mmap_mode="r")

        elif isinstance(labels_source, list):
            assert all([hasattr(timestamp, "__len__") for timestamp in labels_source])
            assert all([len(timestamp) == 2 for timestamp in labels_source])
            labels_source = np.array(labels_source)

        assert isinstance(labels_source, np.ndarray)
        assert labels_source.ndim == 2 and labels_source.shape[1] == 2

        self.shard_duration = shard_duration
        self.shard_count = shard_count
        self.labels_source = labels_source

    def __iter__(self):
        for i in range(self.shard_count):
            labels = []
            shard_start = i * self.shard_duration
            shard_end = shard_start + self.shard_duration
            for start, end in self.labels_source:
                start_in = end > shard_start >= start
                end_in = end > shard_end >= start

                if start_in and end_in:
                    labels = [0.0, 1.0]
                    break

                timestamps_in_shard = (shard_end >= start >= shard_start and shard_end >= end >= shard_start)
                if start_in or end_in or timestamps_in_shard:

                    if start_in:
                        label_start = 0.0
                    else:
                        label_start = inverse_lerp(shard_start, shard_end, start)

                    if end_in:
                        label_end = 1.0
                    else:
                        label_end = inverse_lerp(shard_start, shard_end, end)

                    labels += [label_start, label_end]

            yield labels


def inverse_lerp(start, end, x):
    return (x - start) / (end - start)
