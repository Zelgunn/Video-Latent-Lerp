from typing import Union


class SingleValueLabelsBuilder(object):
    def __init__(self,
                 labels_source: Union[bool, int, float],
                 shard_count: int,
                 ):
        self.shard_count = shard_count

        if isinstance(labels_source, int) or \
                isinstance(labels_source, float):
            labels_source = labels_source >= 1

        self.labels_source = labels_source

    def __iter__(self):
        labels = [0.0, 1.0] if self.labels_source else [0.0, 0.0]
        for i in range(self.shard_count):
            yield labels
