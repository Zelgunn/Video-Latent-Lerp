import os
from typing import Tuple, List, Union, Optional

from modalities import ModalityCollection, RawVideo
from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.data_readers import VideoReader


class UCSDTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 video_buffer_frame_size: Tuple[int, int],
                 verbose=1):
        super(UCSDTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                  shard_duration=shard_duration,
                                                  video_frequency=video_frequency,
                                                  audio_frequency=None,
                                                  modalities=modalities,
                                                  labels_frequency=video_frequency,
                                                  video_buffer_frame_size=video_buffer_frame_size,
                                                  verbose=verbose)
        self.video_frame_size = video_frame_size

    def get_sample_count(self, subset: str) -> int:
        subset_folder = os.path.join(self.dataset_path, subset)
        folders = os.listdir(subset_folder)
        folders = [folder for folder in folders
                   if os.path.isdir(os.path.join(subset_folder, folder))
                   and not folder.endswith("_gt")]
        return len(folders)

    def get_data_sources(self) -> List[DataSource]:
        subsets_names = ("Test", "Train")
        subsets_lengths = {subset_name: self.get_sample_count(subset_name) for subset_name in subsets_names}
        subsets = {}

        labels = self.get_labels(subsets_lengths["Test"])

        for subset in subsets_lengths:
            samples = []
            for i in range(subsets_lengths[subset]):
                filename = "{subset}/{subset}{index:03d}".format(subset=subset, index=i + 1)
                path = os.path.join(self.dataset_path, filename)
                path = os.path.normpath(path)

                if subset == "Test":
                    sample_labels = labels[i]
                else:
                    sample_labels = False

                sample = (path, sample_labels)
                samples.append(sample)
            subsets[subset] = samples

        data_sources = [DataSource(labels_source=sample_labels,
                                   target_path=path,
                                   subset_name=subset,
                                   video_source=VideoReader(path),
                                   video_frame_size=self.video_frame_size)
                        for subset in subsets
                        for path, sample_labels in subsets[subset]]
        return data_sources

    # region Labels
    def get_labels(self, test_sample_count: int):
        if test_sample_count == 36:
            timestamps = self.ped1_anomaly_timestamps()
        else:
            timestamps = self.ped2_anomaly_timestamps()
        return self.convert_labels(timestamps)

    @staticmethod
    def ped1_anomaly_timestamps():
        """
        Source : UCSDped1.m
        """
        base_labels = [
            [(60, 152)],
            [(50, 175)],
            [(91, 200)],
            [(31, 168)],
            [(5, 90), (140, 200)],
            [(1, 100), (110, 200)],
            [(1, 175)],
            [(1, 94)],
            [(1, 48)],
            [(1, 140)],
            [(70, 165)],
            [(130, 200)],
            [(1, 156)],
            [(1, 200)],
            [(138, 200)],
            [(123, 200)],
            [(1, 47)],
            [(54, 120)],
            [(64, 138)],
            [(45, 175)],
            [(31, 200)],
            [(16, 107)],
            [(8, 165)],
            [(50, 171)],
            [(40, 135)],
            [(77, 144)],
            [(10, 122)],
            [(105, 200)],
            [(1, 15), (45, 113)],
            [(175, 200)],
            [(1, 180)],
            [(1, 52), (65, 115)],
            [(5, 165)],
            [(1, 121)],
            [(86, 200)],
            [(15, 108)],
        ]
        return base_labels

    @staticmethod
    def ped2_anomaly_timestamps():
        """
        Source : UCSDped2.m
        """
        base_labels = [
            [(61, 180)],
            [(95, 180)],
            [(1, 146)],
            [(31, 180)],
            [(1, 129)],
            [(1, 159)],
            [(46, 180)],
            [(1, 180)],
            [(1, 120)],
            [(1, 150)],
            [(1, 180)],
            [(88, 180)],
        ]
        return base_labels

    def convert_labels(self, base_labels: List[List[Tuple[int, int]]]) -> List[List[Tuple[float, float]]]:
        labels = []
        for base_sample_labels in base_labels:
            sample_labels = []
            for start, end in base_sample_labels:
                start = (start - 1) / self.video_frequency
                end = (end - 1) / self.video_frequency
                sample_labels.append((start, end))
            labels.append(sample_labels)

        return labels
    # endregion


if __name__ == "__main__":
    ucsd_tf_record_builder = UCSDTFRecordBuilder(dataset_path="../datasets/ucsd/ped1",
                                                 shard_duration=5.0,
                                                 video_frequency=10,
                                                 modalities=ModalityCollection(
                                                     [
                                                         RawVideo(),
                                                     ]
                                                 ),
                                                 video_frame_size=(128, 128),
                                                 video_buffer_frame_size=(128, 128),
                                                 )
    ucsd_tf_record_builder.build()
