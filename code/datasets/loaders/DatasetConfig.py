import json
import os
from typing import Dict, List, Any, Tuple, Union

from datasets.tfrecord_builders import tfrecords_config_filename
from modalities import Modality, ModalityCollection, Pattern
from modalities import RawVideo, OpticalFlow, DoG
from modalities import RawAudio
from misc_utils.misc_utils import int_ceil


def get_shard_count(sample_length: int,
                    shard_size: int
                    ) -> int:
    shard_count = 1 + int_ceil((sample_length - 1) / shard_size)
    return max(2, shard_count)


class DatasetConfig(object):
    def __init__(self,
                 tfrecords_config_folder: str,
                 output_range: Tuple[float, float],
                 ):
        if not os.path.exists(tfrecords_config_folder):
            raise RuntimeError("Folder `{}` does not exist. Check the location of your dataset."
                               .format(tfrecords_config_folder))

        self.tfrecords_config_folder = tfrecords_config_folder
        tf_records_config_filepath = os.path.join(tfrecords_config_folder, tfrecords_config_filename)

        if not os.path.exists(tf_records_config_filepath):
            raise RuntimeError("Could not find file `{}` in `{}`. Your dataset likely has not been built yet. "
                               "Check `datasets/tfrecord_builders` in this project to build your dataset.".
                               format(tfrecords_config_filename, tfrecords_config_folder))

        with open(tf_records_config_filepath, 'r') as file:
            self.tfrecords_config: Dict[str, Any] = json.load(file)

        self.modalities = ModalityCollection.from_config(self.tfrecords_config["modalities"])
        self.subsets: Dict[str, List[str]] = self.tfrecords_config["subsets"]
        self.shard_duration = float(self.tfrecords_config["shard_duration"])
        self.video_frequency = self.tfrecords_config["video_frequency"]
        self.audio_frequency = self.tfrecords_config["audio_frequency"]
        self.max_labels_size: int = int(self.tfrecords_config["max_labels_size"])

        self.modalities_ranges = self.tfrecords_config["modalities_ranges"]

        self.output_range = output_range

    def list_subset_tfrecords(self,
                              subset_name: str
                              ) -> Dict[str, List[str]]:
        subset_files = {}
        subset = self.subsets[subset_name]
        for folder in subset:
            folder = os.path.join(self.tfrecords_config_folder, folder)
            folder = os.path.normpath(folder)
            files = [file for file in os.listdir(folder) if file.endswith(".tfrecord")]
            subset_files[folder] = files
        return subset_files

    def get_modality_shard_size(self,
                                modality: Modality
                                ) -> Union[float, int]:

        if isinstance(modality, (RawVideo, DoG)):
            shard_size = self.video_frequency * self.shard_duration
        elif isinstance(modality, OpticalFlow):
            shard_size = self.video_frequency * self.shard_duration - 1
        elif isinstance(modality, RawAudio):
            shard_size = self.audio_frequency * self.shard_duration
        else:
            raise NotImplementedError(modality.id())

        return shard_size

    def get_modality_max_shard_size(self, modality: Modality) -> int:
        return int_ceil(self.get_modality_shard_size(modality))

    def compute_shards_per_sample(self, pattern: Pattern) -> int:
        shard_counts = []
        for modality_load_info in pattern.flattened:
            if isinstance(modality_load_info, str):
                continue

            sample_length = modality_load_info.length
            modality_type = modality_load_info.modality

            modality = self.modalities[modality_type]
            shard_size = self.get_modality_max_shard_size(modality)
            shard_count = get_shard_count(sample_length, shard_size)

            shard_counts.append(shard_count)

        shards_per_sample: int = max(shard_counts)
        return shards_per_sample
