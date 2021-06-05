import tensorflow as tf
import numpy as np
import os
import json
import time
import datetime
from multiprocessing import Pool
from typing import Union, Tuple, List, Dict, Type, Optional

from modalities import Modality, ModalityCollection
from modalities.utils import float_list_feature
from datasets.modality_builders import ModalityBuilder, VideoBuilder, BuildersList
from datasets.data_readers.VideoReader import VideoReaderProto
from datasets.labels_builders import LabelsBuilder


class DataSource(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str], bool, int, float, List[Tuple[float, float]]],
                 target_path: str,
                 subset_name: str,
                 video_source: Union[str, np.ndarray, List[str], VideoReaderProto] = None,
                 video_frame_size: Tuple[int, int] = None
                 ):
        self.labels_source = labels_source
        self.target_path = target_path
        self.subset_name = subset_name

        self.video_source = video_source
        self.video_frame_size = video_frame_size


tfrecords_config_filename = "tfrecords_config.json"


class TFRecordBuilder(object):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 audio_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 labels_frequency: Union[int, float] = None,
                 video_buffer_frame_size: Tuple[int, int] = None,
                 verbose=1):

        self.dataset_path = dataset_path
        self.shard_duration = shard_duration
        if video_frequency is None and audio_frequency is None:
            raise ValueError("You must specify at least the frequency for either Video or Audio, got None and None")
        self.video_frequency = video_frequency
        self.audio_frequency = audio_frequency
        self.labels_frequency = labels_frequency
        self.modalities = modalities
        self.video_buffer_frame_size = video_buffer_frame_size
        self.verbose = verbose

    def get_data_sources(self) -> List[DataSource]:
        raise NotImplementedError("`get_dataset_sources` should be defined for subclasses.")

    def build(self, core_count=6):
        data_sources = self.get_data_sources()

        subsets_dict: Dict[str, Union[List[str], Dict]] = {}

        min_values = None
        max_values = None
        max_labels_sizes = []

        data_sources_count = len(data_sources)
        start_time = time.time()

        builder_pool = Pool(core_count)
        builders = []

        for data_source in data_sources:
            # region Fill subsets_dict with folders containing shards
            target_path = os.path.relpath(data_source.target_path, self.dataset_path)
            if data_source.subset_name in subsets_dict:
                subsets_dict[data_source.subset_name].append(target_path)
            else:
                subsets_dict[data_source.subset_name] = [target_path]
            # endregion

            builder = builder_pool.apply_async(self.build_one, (data_source,))
            builders.append(builder)

        working_builders = builders
        while len(working_builders) > 0:
            remaining_builders = []

            for builder in working_builders:
                if builder.ready():
                    source_min_values, source_max_values, max_labels_size = builder.get()
                    # region Modalities min/max for normalization (step 1 : get)
                    if min_values is None:
                        min_values = source_min_values
                        max_values = source_max_values
                    else:
                        for modality_type in source_min_values:
                            min_values[modality_type] += source_min_values[modality_type]
                            max_values[modality_type] += source_max_values[modality_type]
                    # endregion
                    max_labels_sizes.append(max_labels_size)
                else:
                    remaining_builders.append(builder)

            # region Print ETA
            if self.verbose > 0 and len(working_builders) != len(remaining_builders):
                i = len(builders) - len(remaining_builders)
                elapsed_time = time.time() - start_time
                eta = elapsed_time * (data_sources_count / i - 1)
                eta = datetime.timedelta(seconds=np.round(eta))
                print("Building {}/{} - ETA: {}".format(i, data_sources_count, eta))
            # endregion

            time.sleep(10.0)

            working_builders = remaining_builders

        # region Modalities min/max for normalization (step 2 : compute)
        modalities_ranges = {
            modality_type.id(): [float(min(min_values[modality_type])),
                                 float(max(max_values[modality_type]))]
            for modality_type in min_values
        }
        # endregion
        max_labels_size = max(max_labels_sizes)

        tfrecords_config = {
            "modalities": self.modalities.get_config(),
            "shard_duration": self.shard_duration,
            "video_frequency": self.video_frequency,
            "audio_frequency": self.audio_frequency,
            "subsets": subsets_dict,
            "modalities_ranges": modalities_ranges,
            "max_labels_size": max_labels_size,
        }

        # TODO : Merge previous tfrecords_config with new when adding new modalities
        with open(os.path.join(self.dataset_path, tfrecords_config_filename), 'w') as file:
            json.dump(tfrecords_config, file)

    def build_one(self, data_source: Union[DataSource, List[DataSource]]):
        builders = self.make_builders(video_source=data_source.video_source,
                                      video_frame_size=data_source.video_frame_size,
                                      video_buffer_frame_size=self.video_buffer_frame_size)

        modality_builder = BuildersList(builders=builders)

        # TODO : Delete previous .tfrecords
        shard_count = modality_builder.get_shard_count()
        labels_iterator = LabelsBuilder(data_source.labels_source,
                                        shard_count=shard_count,
                                        shard_duration=self.shard_duration,
                                        frequency=self.labels_frequency)

        source_iterator = zip(modality_builder, labels_iterator)

        min_values = {}
        max_values = {}
        max_labels_size = 0

        for i, shard in enumerate(source_iterator):
            # region Verbose
            # if self.verbose > 0:
            #     print("\r{} : {}/{}".format(data_source.target_path, i + 1, shard_count), end='')
            # sys.stdout.flush()
            # endregion

            modalities, labels = shard
            modalities: Dict[Type[Modality], np.ndarray] = modalities

            for modality_type, modality_value in modalities.items():
                encoded_features = modality_type.encode_to_tfrecord_feature(modality_value)
                self.write_features_to_tfrecord(encoded_features, data_source.target_path, i, modality_type.id())

                # region Get modality min/max for normalization
                modality_min = modality_value.min()
                modality_max = modality_value.max()
                if modality_type not in min_values:
                    min_values[modality_type] = [modality_min]
                    max_values[modality_type] = [modality_max]
                else:
                    min_values[modality_type] += [modality_min]
                    max_values[modality_type] += [modality_max]
                # endregion

            max_labels_size = max(max_labels_size, len(labels))
            features = {"labels": float_list_feature(labels)}
            self.write_features_to_tfrecord(features, data_source.target_path, i, "labels")

        if self.verbose > 0:
            print("\r{} : Done".format(data_source.target_path))

        return min_values, max_values, max_labels_size

    @staticmethod
    def write_features_to_tfrecord(features: Dict, base_filepath: str, index: int, sub_folder: str = None):
        example = tf.train.Example(features=tf.train.Features(feature=features))
        if sub_folder is not None:
            base_filepath = os.path.join(base_filepath, sub_folder)
        if not os.path.exists(base_filepath):
            os.makedirs(base_filepath)
        filepath = os.path.join(base_filepath, "shard_{:05d}.tfrecord".format(index))
        writer = tf.io.TFRecordWriter(filepath)
        writer.write(example.SerializeToString())

    def make_builders(self,
                      video_source: Union[str, np.ndarray, List[str], VideoReaderProto],
                      video_frame_size: Tuple[int, int],
                      video_buffer_frame_size: Tuple[int, int],
                      ):

        builders: List[ModalityBuilder] = []

        if VideoBuilder.supports_any(self.modalities):
            video_builder = VideoBuilder(shard_duration=self.shard_duration,
                                         source_frequency=self.video_frequency,
                                         modalities=self.modalities,
                                         video_reader=video_source,
                                         default_frame_size=video_frame_size,
                                         buffer_frame_size=video_buffer_frame_size)
            builders.append(video_builder)

        return builders
