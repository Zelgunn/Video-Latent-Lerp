import os
from typing import Tuple, List, Union, Optional, NamedTuple
from enum import IntEnum

from modalities import ModalityCollection, RawVideo
from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.data_readers.VideoReader import VideoReaderProto


class SubwayTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 audio_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 video_buffer_frame_size: Tuple[int, int],
                 version: "SubwayVideo" = None,
                 use_extended_labels=True,
                 verbose=1):
        super(SubwayTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                    shard_duration=shard_duration,
                                                    video_frequency=video_frequency,
                                                    audio_frequency=audio_frequency,
                                                    modalities=modalities,
                                                    video_buffer_frame_size=video_buffer_frame_size,
                                                    verbose=verbose)
        if version is None:
            version = self.guess_version(dataset_path)
            if version is None:
                raise ValueError("Could not determine version from dataset_path {} and version is `None`.".
                                 format(dataset_path))

        self.version = version
        self.video_frame_size = video_frame_size
        self.use_extended_labels = use_extended_labels

    @staticmethod
    def guess_version(dataset_path: str) -> Optional["SubwayVideo"]:
        known_alias = {
            "exit": SubwayVideo.EXIT,
            "entrance": SubwayVideo.ENTRANCE,
            "mall1": SubwayVideo.MALL1,
            "mall2": SubwayVideo.MALL2,
            "mall3": SubwayVideo.MALL3,
        }

        for alias in known_alias:
            if alias in dataset_path:
                return known_alias[alias]

        return None

    def get_data_sources(self) -> List[DataSource]:
        video_filepath = os.path.join(self.dataset_path, self.video_config.video_filename)

        # region train_data_sources
        train_data_sources = []
        train_target_path = os.path.join(self.dataset_path, "Train")
        for start, end in self.video_config.get_train_splits(self.use_extended_labels):
            train_video_reader = VideoReaderProto(video_filepath, start=start, end=end)
            train_labels = False
            target_path = os.path.join(train_target_path, "{}_to_{}".format(start, end))
            if not os.path.isdir(target_path):
                os.makedirs(target_path)
            train_data_source = DataSource(labels_source=train_labels,
                                           target_path=target_path,
                                           subset_name="Train",
                                           video_source=train_video_reader,
                                           video_frame_size=self.video_frame_size)
            train_data_sources.append(train_data_source)
        # endregion

        # region test_data_source
        test_video_reader = VideoReaderProto(video_filepath,
                                             start=self.video_config.training_frames,
                                             end=self.video_config.testing_frames
                                             )
        test_labels = self.video_config.get_anomaly_timestamps_in_seconds(self.use_extended_labels)
        test_target_path = os.path.join(self.dataset_path, "Test")
        if not os.path.isdir(test_target_path):
            os.makedirs(test_target_path)
        test_data_source = DataSource(labels_source=test_labels,
                                      target_path=test_target_path,
                                      subset_name="Test",
                                      video_source=test_video_reader,
                                      video_frame_size=self.video_frame_size)
        # endregion

        data_sources = [*train_data_sources, test_data_source]
        return data_sources

    @property
    def video_config(self) -> "SubwayVideoConfig":
        return known_subway_configs[self.version]


class SubwayVideo(IntEnum):
    EXIT = 0
    ENTRANCE = 1
    MALL1 = 2
    MALL2 = 3
    MALL3 = 4


class SubwayVideoConfig(NamedTuple):
    video_filename: str
    training_minutes: float
    fps: int
    anomaly_timestamps: List[Tuple[int, int]]
    extended_anomaly_timestamps: List[Tuple[int, int]] = None
    testing_minutes: float = None

    @property
    def training_frames(self) -> int:
        return int(self.fps * self.training_minutes * 60)

    @property
    def testing_frames(self) -> Optional[int]:
        if self.testing_minutes is None:
            return None
        return int(self.fps * (self.testing_minutes + self.training_minutes) * 60)

    def get_anomaly_timestamps(self, use_extended: bool):
        if use_extended and self.extended_anomaly_timestamps is None:
            use_extended = False

        return self.extended_anomaly_timestamps if use_extended else self.anomaly_timestamps

    def get_anomaly_timestamps_in_seconds(self, use_extended: bool) -> List[Tuple[float, float]]:
        in_seconds = [
            (
                (start - self.training_frames) / self.fps,
                (end - self.training_frames) / self.fps
            )
            for start, end in self.get_anomaly_timestamps(use_extended)
        ]
        return in_seconds

    def get_train_splits(self, use_extended: bool):
        anomaly_timestamps = self.get_anomaly_timestamps(use_extended)
        anomaly_splits = []
        previous_end = 0
        for start, end in anomaly_timestamps:
            if start < self.training_frames:
                split = (previous_end, start - 1)
                anomaly_splits.append(split)
                previous_end = end
            else:
                break

        if len(anomaly_splits) == 0:
            anomaly_splits = [0, self.training_frames - 1]

        splits = []
        for start, end in anomaly_splits:
            while (end - start) > 1500:
                splits.append((start, start + 1000))
                start += 1000
            splits.append((start, end))

        return splits


# region Pre-defined subsets configurations
exit_config = SubwayVideoConfig(
    video_filename="subway_exit_turnstiles.AVI",
    training_minutes=10.0,
    fps=25,
    anomaly_timestamps=[(40880, 41160), (41400, 41700), (50410, 50710), (50980, 51250), (60160, 60940)]
)

original_entrance_anomaly_timestamps = [
    (27900, 28000), (29750, 29850), (39465, 39565), (67700, 67900), (69240, 69340), (69700, 70000), (72095, 72165),
    (73025, 73075), (73750, 74050), (83415, 83485), (84315, 84400), (85780, 85880), (86475, 86540), (88500, 88640),
    (89720, 89800), (95285, 95385), (96715, 96755),
    (100200, 100425), (115470, 115525), (115800, 115970), (116200, 116225), (117580, 117610), (117760, 117900),
    (118235, 118270), (118700, 119100), (119285, 119300), (124700, 124850), (128025, 128100), (130480, 130675),
]

ours_entrance_anomaly_timestamps = [
    (4120, 5450), (27700, 29850),
    (39400, 39575), (67675, 67925), (67975, 68050),
    (69000, 69350), (69550, 70050), (72050, 72200), (73000, 73100), (73600, 74050), (83050, 83525),
    (84150, 84400), (85325, 85375), (85700, 86025), (86200, 86250), (86350, 86550), (88400, 88900),
    (89550, 89825), (95250, 95400), (96450, 96550), (96675, 96750), (99825, 99975),
    (100050, 100450), (115425, 116000), (117550, 117900), (118200, 118275), (118600, 119075),
    (119250, 119325), (124650, 124850), (128000, 128100), (130375, 130725)
]

entrance_config = SubwayVideoConfig(
    video_filename="subway_entrance_turnstiles.AVI",
    training_minutes=20.0,
    testing_minutes=75.5,
    fps=25,
    anomaly_timestamps=original_entrance_anomaly_timestamps,
    extended_anomaly_timestamps=[
        (2175, 2650), (4120, 5560),
        (12820, 13025), (16120, 16530), (17020, 17650),
        (20130, 20620), (21680, 21840), (27750, 29950), (33350, 35415), (39390, 39575),
        (41000, 41150), (41350, 41500), (44775, 44975), (45075, 47000), (52275, 53425),
        (57375, 57500), (59800, 59900), (67450, 68050), (68950, 69350), (69520, 70200),
        (70770, 72200), (73000, 73075), (73450, 74110), (80300, 80500), (79420, 80510),
        (81130, 81770), (82210, 82415), (82490, 87150), (87300, 88940), (89570, 89870),
        (90000, 91000), (92025, 92075), (92200, 92375), (92450, 93690), (94255, 95450),
        (96675, 96750),
        (100050, 100575), (106125, 106575), (111300, 111645), (114895, 115025),
        (115450, 115550), (115735, 116025), (116110, 116525), (117465, 117625),
        (117675, 117950), (118135, 118275), (118585, 119125), (119250, 119325),
        (124645, 124850), (128000, 128125), (130350, 130725)
    ]
)

mall1_config = SubwayVideoConfig(
    video_filename="mall_1.AVI",
    training_minutes=20.0,
    fps=25,
    anomaly_timestamps=[
        (68700, 68800), (69320, 69440), (70035, 70200), (70400, 70515), (70650, 70760), (71350, 71460), (71910, 72000),
        (72450, 72550), (74020, 74110), (74410, 74500), (75935, 76010), (76650, 76750), (77225, 77300), (77750, 77830),
        (78280, 78370), (78825, 78920), (79205, 79300), (79585, 79700), (80625, 80700), (81250, 81320), (82140, 82235)
    ],
)

mall2_config = SubwayVideoConfig(
    video_filename="mall_2.AVI",
    training_minutes=20.0,
    fps=25,
    anomaly_timestamps=[
        (6040, 6170),
        (33155, 33230), (44600, 44560), (47570, 44660), (48020, 48130), (48640, 48740), (49090, 49175), (49225, 49300),
        (50090, 50190), (50840, 50915), (51220, 51300), (51605, 51680), (51940, 52020), (52310, 52415), (52740, 52815),
        (53145, 53200), (53770, 53850)
    ],
)

mall3_config = SubwayVideoConfig(
    video_filename="mall_3.AVI",
    training_minutes=9.0,
    fps=25,
    anomaly_timestamps=[
        (14000, 14030), (18280, 18350), (29095, 29135), (34290, 34425), (34750, 34820), (35275, 35400), (35730, 35790),
        (36340, 36420), (36940, 37000), (37780, 37850), (38200, 38260), (38680, 38715), (38950, 39000), (39250, 39310),
        (39610, 39650), (41210, 41250), (41775, 41850), (42120, 42160), (42470, 42520), (61515, 61570), (69075, 69135)
    ],
)

known_subway_configs = {
    SubwayVideo.EXIT: exit_config,
    SubwayVideo.ENTRANCE: entrance_config,
    SubwayVideo.MALL1: mall1_config,
    SubwayVideo.MALL2: mall2_config,
    SubwayVideo.MALL3: mall3_config,
}
# endregion


if __name__ == "__main__":
    subway_tf_record_builder = SubwayTFRecordBuilder(dataset_path="../datasets/subway/entrance",
                                                     shard_duration=1.28,
                                                     video_frequency=25,
                                                     audio_frequency=None,
                                                     modalities=ModalityCollection(
                                                         [
                                                             RawVideo(),
                                                         ]
                                                     ),
                                                     video_frame_size=(160, 160),
                                                     video_buffer_frame_size=(160, 160),
                                                     use_extended_labels=True,
                                                     )
    subway_tf_record_builder.build()
