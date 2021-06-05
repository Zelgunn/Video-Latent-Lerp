import os
import scipy.io as sio
import numpy as np
from typing import Tuple, List, Union, Optional

from modalities import ModalityCollection, RawVideo
from datasets.tfrecord_builders import TFRecordBuilder, DataSource


class AvenueTFRB(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 video_buffer_frame_size: Tuple[int, int],
                 verbose=1):
        super(AvenueTFRB, self).__init__(dataset_path=dataset_path,
                                         shard_duration=shard_duration,
                                         video_frequency=video_frequency,
                                         audio_frequency=None,
                                         modalities=modalities,
                                         labels_frequency=video_frequency,
                                         video_buffer_frame_size=video_buffer_frame_size,
                                         verbose=verbose)
        self.video_frame_size = video_frame_size

    def get_data_sources(self) -> List[DataSource]:
        subsets_lengths = {"testing": 21, "training": 16}
        subsets_targets = {"testing": "Test", "training": "Train"}
        subsets = {}

        def to_path(filename):
            return os.path.normpath(os.path.join(self.dataset_path, filename))

        def load_labels(path):
            mat_contents = sio.loadmat(path)
            mat_labels = mat_contents["volLabel"]
            mat_labels = np.stack(mat_labels[0], axis=0)
            return mat_labels

        for subset in subsets_lengths:
            subset_data = []
            for i in range(subsets_lengths[subset]):
                target_path = to_path("{target}/{index:02d}").format(target=subsets_targets[subset], index=i + 1)
                video_path = to_path("{subset}_videos/{index:02d}.avi".format(subset=subset, index=i + 1))

                if subsets_targets[subset] == "Test":
                    labels_path = to_path("ground_truth_demo/testing_label_mask/{index}_label.mat"
                                          .format(subset=subset, index=i + 1))
                    labels = load_labels(labels_path)
                else:
                    labels = False

                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                sample_data = (target_path, video_path, labels)
                subset_data.append(sample_data)
            subsets[subsets_targets[subset]] = subset_data

        data_sources = [DataSource(labels_source=labels,
                                   target_path=target_path,
                                   subset_name=subset,
                                   video_source=video_path,
                                   video_frame_size=self.video_frame_size)
                        for subset in subsets
                        for target_path, video_path, labels in subsets[subset]]
        return data_sources


if __name__ == "__main__":
    avenue_tf_record_builder = AvenueTFRB(dataset_path="../datasets/avenue",
                                          shard_duration=1.28,
                                          video_frequency=25,
                                          modalities=ModalityCollection(
                                              [
                                                  RawVideo(),
                                              ]
                                          ),
                                          video_frame_size=(128, 128),
                                          video_buffer_frame_size=(128, 128),
                                          )
    avenue_tf_record_builder.build()
