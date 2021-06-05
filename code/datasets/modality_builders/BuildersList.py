from typing import List

from datasets.modality_builders import ModalityBuilder
from misc_utils.misc_utils import int_ceil


class BuildersList(object):
    def __init__(self, builders: List[ModalityBuilder]):
        modalities = []
        for builder in builders:
            for modality in builder.modalities:
                modalities.append(modality)

        self.builders: List[ModalityBuilder] = builders

    def __iter__(self):
        builders_iterator = zip(*self.builders)
        for partial_shards in builders_iterator:
            shard = {modality: partial_shard[modality]
                     for partial_shard in partial_shards
                     if partial_shard is not None
                     for modality in partial_shard}

            yield shard

    def get_shard_count(self) -> int:
        min_shard_count = None

        for builder in self.builders:
            total_duration = builder.source_frame_count / builder.source_frequency
            shard_count = total_duration / builder.shard_duration
            if min_shard_count is None:
                min_shard_count = shard_count
            else:
                min_shard_count = min(min_shard_count, shard_count)

        min_shard_count = int_ceil(min_shard_count)
        return min_shard_count
