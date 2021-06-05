import os
from typing import Tuple, List, Union, Optional

from modalities import ModalityCollection, RawVideo
from datasets.tfrecord_builders import TFRecordBuilder, DataSource


class ShanghaiTechTFRB(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 video_buffer_frame_size: Tuple[int, int],
                 verbose=1):
        super(ShanghaiTechTFRB, self).__init__(dataset_path=dataset_path,
                                               shard_duration=shard_duration,
                                               video_frequency=video_frequency,
                                               audio_frequency=None,
                                               modalities=modalities,
                                               labels_frequency=video_frequency,
                                               video_buffer_frame_size=video_buffer_frame_size,
                                               verbose=verbose)
        self.video_frame_size = video_frame_size

        self.train_videos_folder = os.path.join(self.dataset_path, "training/videos")
        self.test_frames_root_folder = os.path.join(self.dataset_path, "testing/frames")
        self.labels_folder = os.path.join(self.dataset_path, "testing/test_frame_mask")

    def get_data_sources(self) -> List[DataSource]:
        return self.get_train_data_sources() + self.get_test_data_sources()

    def get_train_data_sources(self) -> List[DataSource]:
        videos_filenames = self.list_training_videos_filenames()

        train_folder = os.path.join(self.dataset_path, "Train")
        subset = [(os.path.join(train_folder, filename[:-4]), os.path.join(self.train_videos_folder, filename))
                  for filename in videos_filenames]

        data_sources = [DataSource(labels_source=False,
                                   target_path=target_path,
                                   subset_name="Train",
                                   video_source=video_path,
                                   video_frame_size=self.video_frame_size)
                        for target_path, video_path in subset]
        return data_sources

    def get_test_data_sources(self) -> List[DataSource]:
        frames_folders = self.list_testing_frames_folders()

        test_folder = os.path.join(self.dataset_path, "Test")
        subset = [(
            os.path.join(test_folder, folder),
            os.path.join(self.test_frames_root_folder, folder),
            os.path.join(self.labels_folder, "{}.npy".format(folder))
        )
            for folder in frames_folders]

        data_sources = [DataSource(labels_source=labels,
                                   target_path=target_path,
                                   subset_name="Test",
                                   video_source=video_path,
                                   video_frame_size=self.video_frame_size)
                        for target_path, video_path, labels in subset]
        return data_sources

    def list_training_videos_filenames(self) -> List[str]:
        elements = os.listdir(self.train_videos_folder)
        videos_filenames = []
        for element in elements:
            if element.endswith(".avi") and os.path.isfile(os.path.join(self.train_videos_folder, element)):
                videos_filenames.append(element)

        return videos_filenames

    def list_testing_frames_folders(self) -> List[str]:
        elements = os.listdir(self.test_frames_root_folder)
        frames_folders = []
        for element in elements:
            if os.path.isdir(os.path.join(self.test_frames_root_folder, element)):
                frames_folders.append(element)
        return frames_folders


if __name__ == "__main__":
    avenue_tf_record_builder = ShanghaiTechTFRB(dataset_path="../datasets/shanghaitech",
                                                shard_duration=1.28,
                                                video_frequency=25,
                                                modalities=ModalityCollection(
                                                    [
                                                        RawVideo(),
                                                    ]
                                                ),
                                                video_frame_size=(256, 256),
                                                video_buffer_frame_size=(256, 256)
                                                )
    avenue_tf_record_builder.build()
