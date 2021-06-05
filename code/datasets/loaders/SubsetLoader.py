import tensorflow as tf
import numpy as np
import os
import copy
import random
from typing import Dict, Tuple, Optional, List

from datasets.loaders import DatasetConfig
from modalities import Modality, ModalityCollection, Pattern
from modalities import RawVideo
from misc_utils.misc_utils import int_ceil, int_floor


class SubsetLoader(object):
    # region Initialization
    def __init__(self,
                 config: DatasetConfig,
                 subset_name: str):
        self.config = config
        self.check_unsupported_shard_sizes()

        self.subset_name = subset_name
        self.subset_files = {folder: sorted(files)
                             for folder, files in config.list_subset_tfrecords(subset_name).items()}
        self.subset_folders = list(self.subset_files.keys())

        self._train_tf_dataset: Optional[tf.data.Dataset] = None
        self._test_tf_dataset: Optional[tf.data.Dataset] = None

    def check_unsupported_shard_sizes(self):
        for modality in self.config.modalities:

            shard_size = self.config.get_modality_shard_size(modality)
            max_shard_size = int_ceil(shard_size)
            initial_shard_size = int_floor(shard_size)
            if max_shard_size != initial_shard_size:
                raise ValueError("max_shard_size != initial_shard_size : "
                                 "SubsetLoader doesn't support this case yet.")

    # endregion

    # region Make tf.data.Dataset(s)
    def make_tf_dataset(self,
                        pattern: Pattern,
                        subset_folders: List[str] = None,
                        ) -> tf.data.Dataset:
        if subset_folders is None:
            subset_folders = self.subset_folders
        subset_folders = copy.copy(subset_folders)

        k = os.cpu_count()
        shards_per_sample = self.config.compute_shards_per_sample(pattern)

        generator = self.make_shard_filepath_generator(subset_folders, pattern, shards_per_sample)
        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_types=tf.string,
                                                 output_shapes=())
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=k)
        dataset = dataset.batch(pattern.modalities_per_sample).prefetch(1)

        dataset = dataset.map(lambda serialized_shards: self.parse_shard(serialized_shards, pattern),
                              num_parallel_calls=k)

        dataset = dataset.batch(shards_per_sample)
        dataset = dataset.map(lambda shards, shards_sizes: self.join_shards_and_extract_one_random(shards,
                                                                                                   shards_sizes,
                                                                                                   pattern),
                              num_parallel_calls=k)

        dataset = dataset.map(self.normalize_modalities, num_parallel_calls=k)
        dataset = dataset.map(pattern.apply, num_parallel_calls=k)

        return dataset

    def make_tf_datasets_splits(self,
                                pattern: Pattern,
                                split: float,
                                subset_folders: List[str] = None,
                                ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        if split <= 0.0 or split >= 1.0:
            raise ValueError("Split must be strictly between 0.0 and 1.0, found {}.".format(split))

        if subset_folders is None:
            subset_folders = self.subset_folders
        subset_folders = copy.copy(subset_folders)

        if len(subset_folders) == 1:
            return self.make_tf_dataset(pattern, subset_folders), None

        train_count = int_ceil(len(subset_folders) * split)
        random.shuffle(subset_folders)

        if train_count == len(subset_folders):
            train_count = len(subset_folders) - 1

        train_dataset = self.make_tf_dataset(pattern, subset_folders[:train_count])
        validation_dataset = self.make_tf_dataset(pattern, subset_folders[train_count:])
        return train_dataset, validation_dataset

    def make_source_browser(self,
                            pattern: Pattern,
                            source_index: int,
                            stride: int) -> tf.data.Dataset:
        shards_per_sample: int = self.config.compute_shards_per_sample(pattern)
        source_folder = self.subset_folders[source_index]
        generator = self.make_browser_filepath_generator(source_folder, shards_per_sample, pattern)
        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_types=tf.string,
                                                 output_shapes=())

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.batch(pattern.modalities_per_sample).prefetch(1)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, pattern))

        dataset = dataset.batch(shards_per_sample)
        dataset = dataset.map(lambda shards, shard_sizes:
                              self.join_shards_and_extract_all(shards, shard_sizes, pattern, stride))

        dataset = dataset.unbatch()
        dataset = dataset.map(self.normalize_modalities)
        dataset = dataset.map(pattern.apply)
        dataset = dataset.batch(1)

        return dataset

    # endregion

    # region 1) Generate filepaths
    @staticmethod
    def make_shard_filepath_generator(folders: List[str],
                                      pattern: Pattern,
                                      shards_per_sample: int):
        modality_ids = list(pattern.modality_ids)
        if pattern.contains_labels:
            modality_ids.append("labels")

        def generator():
            # noinspection DuplicatedCode
            while True:
                source_index = np.random.randint(len(folders))
                source_folder = folders[source_index]
                files = []
                shards_count = None
                for modality_id in modality_ids:
                    folder = os.path.join(source_folder, modality_id)

                    modality_files = [os.path.join(folder, file)
                                      for file in os.listdir(folder) if file.endswith(".tfrecord")]

                    files.append(modality_files)
                    if shards_count is None:
                        shards_count = len(modality_files)
                    elif shards_count != len(modality_files):
                        raise ValueError("Modalities don't have the same number of shards in "
                                         "{}.".format(folder))

                offset = np.random.randint(shards_count - shards_per_sample + 1)
                for shard_index in range(offset, offset + shards_per_sample):
                    for file_index in range(len(files)):
                        yield files[file_index][shard_index]

        return generator

    @staticmethod
    def make_browser_filepath_generator(source_folder: str,
                                        shards_per_sample: int,
                                        pattern: Pattern):
        modality_ids = list(pattern.modality_ids) + ["labels"]

        # noinspection DuplicatedCode
        def generator():
            files = []
            shards_count = None
            for modality_id in modality_ids:
                modality_folder = os.path.join(source_folder, modality_id)
                modality_files = [os.path.join(modality_folder, file)
                                  for file in os.listdir(modality_folder) if file.endswith(".tfrecord")]
                files.append(modality_files)

                if shards_count is None:
                    shards_count = len(modality_files)
                elif shards_count != len(modality_files):
                    raise ValueError("Modalities don't have the same number of shards in "
                                     "{}.".format(modality_folder))

            for shard_index in range(shards_count - shards_per_sample + 1):
                for i in range(shards_per_sample):
                    for modality_index in range(len(files)):
                        yield files[modality_index][shard_index + i]

        return generator

    # endregion

    # region 2) Parse shard
    def parse_shard(self, serialized_examples, pattern: Pattern):
        serialized_examples.set_shape(pattern.modalities_per_sample)

        features_decoded, modalities_shard_size = {}, {}

        for i, modality_type in enumerate(pattern.required_modality_types):
            modality = self.modalities[modality_type]
            modality_id = modality.id()
            modality_features = modality.tfrecord_features()
            modality_example = serialized_examples[i]

            parsed_features = tf.io.parse_single_example(modality_example, modality_features)

            decoded_modality = modality.decode_from_tfrecord_feature(parsed_features)
            decoded_modality, modality_size = self.pad_modality_if_needed(modality, decoded_modality)

            features_decoded[modality_id] = decoded_modality
            modalities_shard_size[modality_id] = modality_size

        if pattern.contains_labels:
            labels_features = {"labels": tf.io.VarLenFeature(tf.float32)}
            labels_example = serialized_examples[-1]

            parsed_features = tf.io.parse_single_example(labels_example, labels_features)

            labels = parsed_features["labels"].values
            labels = self.pad_labels_if_needed(labels)
            features_decoded["labels"] = labels

        return features_decoded, modalities_shard_size

    def pad_modality_if_needed(self,
                               modality: Modality,
                               decoded_modality: tf.Tensor
                               ) -> Tuple[tf.Tensor, tf.Tensor]:
        modality_size = tf.shape(decoded_modality)[0]
        modality_max_size = self.config.get_modality_max_shard_size(modality)
        pad_size = modality_max_size - modality_size

        def pad_modality():
            paddings_rank = tf.rank(decoded_modality)
            size_paddings = [[0, pad_size]]
            shape_paddings = tf.zeros(shape=[paddings_rank - 1, 2], dtype=tf.int64)
            paddings = tf.concat([size_paddings, shape_paddings], axis=0,
                                 name=modality.id() + "_paddings")
            return tf.pad(decoded_modality, paddings)

        def identity():
            return decoded_modality

        decoded_modality = tf.cond(pred=pad_size > 0,
                                   true_fn=pad_modality,
                                   false_fn=identity)

        return decoded_modality, modality_size

    def pad_labels_if_needed(self, labels: tf.Tensor):
        labels_size = tf.shape(labels)[0]
        max_labels_size = self.config.max_labels_size
        pad_size = max_labels_size - labels_size

        def pad_labels():
            paddings = [[pad_size, 0]]
            return tf.pad(labels, paddings)

        def identity():
            return labels

        labels = tf.cond(pred=pad_size > 0,
                         true_fn=pad_labels,
                         false_fn=identity)

        return labels

    # endregion

    # region 3) Join shards
    def join_shards_and_extract_one_random(self,
                                           shards: Dict[str, tf.Tensor],
                                           shard_sizes: Dict[str, tf.Tensor],
                                           pattern: Pattern):
        offset = tf.random.uniform(shape=(), minval=0, maxval=1.0, dtype=tf.float32, name="offset")
        return self.join_shards(shards, shard_sizes, offset, pattern)

    def join_shards_and_extract_all(self,
                                    shards: Dict[str, tf.Tensor],
                                    shard_sizes: Dict[str, tf.Tensor],
                                    pattern: Pattern,
                                    stride: int
                                    ):
        with tf.name_scope("join_shards_ordered"):
            reference_modality_id = pattern.modality_ids[0]
            size = shard_sizes[reference_modality_id][0]
            stride = tf.constant(stride, tf.int32, name="stride")
            result_count = size // stride

            def loop_body(i, step_shards_arrays: Dict[str, tf.TensorArray]):
                step_offset = tf.cast(i * stride, tf.float32) / tf.cast(size - 1, tf.float32)
                step_joint_shards = self.join_shards(shards, shard_sizes, step_offset, pattern,
                                                     length_map_function=max)
                for modality_id in step_shards_arrays:
                    modality = step_joint_shards[modality_id]
                    step_shards_arrays[modality_id] = step_shards_arrays[modality_id].write(i, modality)
                i += 1
                return i, step_shards_arrays

            i_initializer = tf.constant(0, tf.int32)
            shards_arrays = {modality_id: tf.TensorArray(dtype=shards[modality_id].dtype, size=result_count)
                             for modality_id in shards}
            results = tf.while_loop(cond=lambda i, _: i < result_count,
                                    body=loop_body,
                                    loop_vars=[i_initializer, shards_arrays],
                                    parallel_iterations=1)
            joint_shard: Dict[str, tf.TensorArray] = results[1]
            joint_shard = {modality_id: joint_shard[modality_id].stack(name="stack_{}".format(modality_id))
                           for modality_id in joint_shard}
            return joint_shard

    def join_shards(self,
                    shards: Dict[str, tf.Tensor],
                    shard_sizes: Dict[str, tf.Tensor],
                    offset: tf.Tensor,
                    pattern: Pattern,
                    length_map_function=max):

        joint_shards = {}
        labels_range = None
        labels_offset = None

        # region for each modality : join shards and extract needed length
        for modality in self.modalities:
            modality_type = type(modality)
            if modality_type not in pattern.required_modality_types:
                continue

            modality_id = modality.id()
            modality_shards = shards[modality_id]
            modality_shard_sizes = shard_sizes[modality_id]

            with tf.name_scope(modality_id):
                total_size = tf.cast(tf.reduce_sum(modality_shard_sizes), tf.int32, name="total_shard_size")

                modality_load_infos = pattern.as_dict[modality_type]
                modality_sample_size = length_map_function([load_info.length for load_info in modality_load_infos])
                modality_sample_size_op = tf.constant(modality_sample_size, name="modality_sample_size")

                # region modality_offset = offset * modality_offset_range
                modality_effective_size = modality_shard_sizes[0]
                modality_offset_range = tf.minimum(modality_effective_size, total_size - modality_sample_size_op)
                modality_offset_range = tf.cast(modality_offset_range, tf.float32)
                modality_offset = tf.cast(offset * modality_offset_range, tf.int32, name="offset")
                # endregion

                # region modality_shards = concatenate(modality_shards)
                modality_shards_shape = tf.shape(modality_shards, name="modality_shard_shape")
                modality_shards_shape.set_shape(modality.rank() + 1)
                modality_shards_shape = tf.unstack(modality_shards_shape, name="unstack_modality_shape")
                shards_per_sample, modality_size, *modality_shape = modality_shards_shape
                modality_shards_shape = [shards_per_sample * modality_size, *modality_shape]
                modality_shards = tf.reshape(modality_shards, modality_shards_shape, "concatenate_shards")
                # endregion

                modality_shards = modality_shards[modality_offset:modality_offset + modality_sample_size_op]
                joint_shards[modality_id] = modality_shards

            if "labels" in shards and labels_range is None:
                labels_range = tf.cast(modality_sample_size_op, tf.float32) / tf.cast(total_size, tf.float32)
                size_ratio = tf.cast(modality_effective_size, tf.float32) / tf.cast(total_size, tf.float32)
                labels_offset = offset * size_ratio
        # endregion

        # region if labels in shards: join labels shards and extract needed length
        if "labels" in shards:
            labels: tf.Tensor = shards["labels"]

            shards_per_sample = tf.cast(shards_per_sample, tf.float32)
            shard_labels_offset = tf.range(shards_per_sample, dtype=tf.float32, name="shard_labels_offset")
            shard_labels_offset = tf.expand_dims(shard_labels_offset, axis=-1)

            labels = (labels + shard_labels_offset) / shards_per_sample
            labels = tf.clip_by_value(labels, labels_offset, labels_offset + labels_range)
            labels = (labels - labels_offset) / labels_range

            labels = tf.reshape(labels, shape=[-1, 2])
            joint_shards["labels"] = labels
        # endregion

        return joint_shards

    # endregion

    # region 4) Normalize modalities
    def normalize_modalities(self, modalities: Dict[str, tf.Tensor]):
        for modality_id in modalities:
            if modality_id == "labels":
                continue

            modality_value = modalities[modality_id]
            if modality_id == RawVideo.id():
                modality_value = modality_value / tf.constant(255.0, modality_value.dtype)
            else:
                modality_min, modality_max = self.config.modalities_ranges[modality_id]
                modality_value = (modality_value - modality_min) / (modality_max - modality_min)
            modality_value *= (self.config.output_range[1] - self.config.output_range[0])
            modality_value += self.config.output_range[0]
            modalities[modality_id] = modality_value
        return modalities

    # endregion

    # region Properties
    @property
    def modalities(self) -> ModalityCollection:
        return self.config.modalities

    @property
    def source_count(self) -> int:
        return len(self.subset_folders)

    # endregion

    # region Utility
    def get_batch(self, batch_size: int, pattern: Pattern):
        dataset = self.make_tf_dataset(pattern)
        dataset = dataset.batch(batch_size)
        results = None
        for results in dataset:
            break

        return results

    @staticmethod
    def timestamps_labels_to_frame_labels(timestamps: np.ndarray, frame_count: int):
        # [batch_size, pairs_count, 2] => [batch_size, frame_count]
        # start, end = timestamps[:,:,0], timestamps[:,:,1]
        batch_size, timestamps_per_sample, _ = timestamps.shape
        epsilon = 1e-4
        starts = timestamps[:, :, 0]
        ends = timestamps[:, :, 1]
        labels_are_not_equal = np.abs(starts - ends) > epsilon  # [batch_size, pairs_count]

        frame_labels = np.empty(shape=[batch_size, frame_count], dtype=np.bool)

        frame_duration = 1.0 / frame_count
        for frame_id in range(frame_count):
            start_time = frame_id / frame_count
            end_time = start_time + frame_duration

            start_in = np.all([start_time >= starts, start_time <= ends], axis=0)
            end_in = np.all([end_time >= starts, end_time <= ends], axis=0)

            frame_in = np.any([start_in, end_in], axis=0)

            frame_in = np.logical_and(frame_in, labels_are_not_equal)
            frame_in = np.any(frame_in, axis=1)

            frame_labels[:, frame_id] = frame_in

        return frame_labels
    # endregion
