import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
from enum import IntEnum
from typing import Union, List, Optional, Iterator, Tuple


class VideoReaderMode(IntEnum):
    CV_VIDEO_CAPTURE = 0,
    NP_ARRAY = 1,
    IMAGE_COLLECTION = 2


class VideoReaderProto(object):
    def __init__(self,
                 video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]],
                 mode: VideoReaderMode = None,
                 start=0, end=None):
        self.video_source = video_source
        self.mode = mode
        self.start = start
        self.end = end

    def to_video_reader(self) -> "VideoReader":
        return VideoReader(video_source=self.video_source,
                           mode=self.mode,
                           start=self.start,
                           end=self.end)


class VideoReader(object):
    def __init__(self,
                 video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]],
                 mode: VideoReaderMode = None,
                 start=0, end=None):

        if mode is None:
            self.mode = infer_video_reader_mode(video_source)
        else:
            assert mode == infer_video_reader_mode(video_source)
            self.mode = mode

        self.video_source = video_source

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_array: Optional[np.array] = None
        self.image_collection: Optional[List[str]] = None

        # region Select & set container
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            if isinstance(video_source, str):
                self.video_capture = cv2.VideoCapture(video_source)

                if not self.video_capture.isOpened():
                    raise RuntimeError("Could not open {}.".format(self.video_source))
            else:
                self.video_capture = video_source
            max_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        elif self.mode == VideoReaderMode.NP_ARRAY:
            if isinstance(video_source, str):
                self.video_array = np.load(video_source, mmap_mode="r")
            elif isinstance(video_source, np.ndarray):
                self.video_array = video_source
            else:
                self.video_array = np.asarray(video_source)
            max_frame_count = len(self.video_array)

        else:
            if isinstance(video_source, str):
                images_names = os.listdir(video_source)
                self.image_collection = [os.path.join(video_source, image_name) for image_name in images_names
                                         if is_image_format_supported(image_name)]
                self.image_collection = sorted(self.image_collection)
            else:
                self.image_collection = video_source
            max_frame_count = len(self.image_collection)
        # endregion

        # region End / Start
        if end is None:
            self.end = max_frame_count
        elif end < 0:
            self.end = max_frame_count + end
        else:
            self.end = min(end, max_frame_count)

        if start is None:
            self.start = 0
        elif start < 0:
            self.start = max_frame_count + start
        else:
            self.start = min(start, max_frame_count)

        if self.end <= self.start:
            raise ValueError("End frame index ({}) is less or equal than"
                             " the start frame index ({}). Max frame count is {}. Mode is {}."
                             .format(self.end, self.start, max_frame_count, self.mode))
        # endregion

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start)

        for i in range(self.start, self.end):
            if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
            elif self.mode == VideoReaderMode.NP_ARRAY:
                frame = self.video_array[i]
            else:
                frame = Image.open(self.image_collection[i])
                frame = np.array(frame)

            yield frame

    # region Properties
    @property
    def frame_count(self):
        return self.end - self.start

    @property
    def frame_height(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1]
        else:
            frame = Image.open(self.image_collection[0])
            return frame.height

    @property
    def frame_width(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[2]
        else:
            frame = Image.open(self.image_collection[0])
            return frame.width

    @property
    def frame_channels(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return 3
        elif self.mode == VideoReaderMode.NP_ARRAY:
            if self.video_array.ndim == 4:
                return self.video_array.shape[3]
            else:
                return 1
        else:
            frame = Image.open(self.image_collection[0])
            frame = np.array(frame)
            if frame.ndim == 3:
                return frame.shape[2]
            else:
                return 1

    @property
    def frame_shape(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return [self.frame_height, self.frame_width, 3]
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1:]
        else:
            frame = Image.open(self.image_collection[0])
            frame = np.array(frame)
            return frame.shape

    @property
    def frame_size(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return [self.frame_height, self.frame_width]
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1:3]
        else:
            frame = Image.open(self.image_collection[0])
            return tuple(reversed(frame.size))
    # endregion

    def rewrite_video(self, target_path: str, fps: int, frame_size: Tuple[int, int] = None):
        if frame_size is None:
            frame_size = (self.frame_width, self.frame_height)

        out = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, frame_size)

        for frame in tqdm(self, total=self.frame_count):
            out.write(frame)
        out.release()


def infer_video_reader_mode(video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]]):
    if isinstance(video_source, cv2.VideoCapture):
        return VideoReaderMode.CV_VIDEO_CAPTURE

    elif isinstance(video_source, np.ndarray):
        assert video_source.ndim == 3 or video_source.ndim == 4
        return VideoReaderMode.NP_ARRAY

    elif isinstance(video_source, list):
        assert all([isinstance(element, str) for element in video_source])
        assert all([os.path.isfile(element) for element in video_source])
        return VideoReaderMode.IMAGE_COLLECTION

    elif not isinstance(video_source, str):
        raise ValueError("\'video_source\' must either be a string, a VideoCapture, a ndarray or a list of strings."
                         "Received `{}` of with type `{}`".format(video_source, type(video_source)))

    elif os.path.isdir(video_source):
        return VideoReaderMode.IMAGE_COLLECTION

    elif os.path.isfile(video_source):
        if ".npy" in video_source or ".npz" in video_source:
            return VideoReaderMode.NP_ARRAY
        else:
            return VideoReaderMode.CV_VIDEO_CAPTURE

    else:
        raise ValueError("\'video_source\' : {} does not exist.".format(video_source))


def is_image_format_supported(image_name: str) -> bool:
    # source : https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html
    image_name = image_name.lower()
    supported_image_formats = [".bmp", ".eps", ".gif", ".icns", ".ico", ".im", ".jpg", ".jpeg", ".msp", ".pcx", ".png",
                               ".ppm", ".sgi", ".spider", ".tif", ".tiff", ".xbm", ".blp", ".cur", ".dcx", ".dds",
                               ".fli", ".flc", ".fpx", ".ftex", ".gbr", ".gd", ".imt", ".iptc", ".naa", ".mcidas",
                               ".mic", ".mpo", ".pcd", ".pixar", ".psd", ".tga", ".wal", ".xpm"]
    for image_format in supported_image_formats:
        if image_name.endswith(image_format):
            return True
    return False


def main():
    video_reader = VideoReader(r"..\datasets\ucsd\ped2\Test\Test001_gt")
    print(video_reader.image_collection)
    for frame in video_reader:
        cv2.imshow("frame", frame)
        cv2.waitKey(40)


if __name__ == "__main__":
    main()
