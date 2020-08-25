import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import cv2
from typing import Union

from callbacks import TensorBoardPlugin
from misc_utils.summary_utils import image_summary
from datasets import SubsetLoader
from modalities import Pattern


class LandmarksVideoCallback(TensorBoardPlugin):
    def __init__(self,
                 subset: Union[SubsetLoader, tf.data.Dataset],
                 autoencoder: Model,
                 tensorboard: TensorBoard,
                 pattern: Pattern = None,
                 epoch_freq=1,
                 output_size=(512, 512),
                 line_thickness=1,
                 fps=25,
                 is_train_callback=False,
                 prefix=""):
        super(LandmarksVideoCallback, self).__init__(tensorboard,
                                                     update_freq="epoch",
                                                     epoch_freq=epoch_freq)

        self.output_size = output_size
        self.line_thickness = line_thickness
        self.fps = fps
        self.autoencoder = autoencoder
        self.writer_name = self.train_run_name if is_train_callback else self.validation_run_name
        self.prefix = prefix if (len(prefix) == 0) else (prefix + "_")
        self.ground_truth_images = None

        if isinstance(subset, SubsetLoader):
            self.inputs, self.outputs = subset.get_batch(batch_size=4, pattern=pattern)
        else:
            for batch in subset.batch(batch_size=4).take(1):
                inputs, outputs = batch
                self.inputs = inputs.numpy()
                self.outputs = outputs.numpy()

    def _write_logs(self, index: int):
        with self._get_writer(self.writer_name).as_default():
            if index == 0:
                self.ground_truth_images = self.landmarks_to_image(self.outputs, color=(255, 0, 0))
                images = tf.convert_to_tensor(self.ground_truth_images)
                image_summary(self.prefix + "ground_truth_landmarks", images, max_outputs=4, step=index, fps=self.fps)

            predicted = self.autoencoder.predict(self.inputs)

            images = self.landmarks_to_image(predicted, color=(0, 255, 0))
            images = tf.convert_to_tensor(images)
            comparison = tf.convert_to_tensor(images + self.ground_truth_images)
            image_summary(self.prefix + "predicted_landmarks", images, max_outputs=4, step=index, fps=self.fps)
            image_summary(self.prefix + "comparison", comparison, max_outputs=4, step=index, fps=self.fps)

    def landmarks_to_image(self, landmarks_batch: np.ndarray, color):
        if len(landmarks_batch.shape) == 3:
            batch_size = landmarks_batch.shape[0]
            landmarks_batch = landmarks_batch.reshape([batch_size, - 1, 68, 2])

        sections = [17, 22, 27, 31, 36, 42, 48, 60]

        batch_size, sequence_length, landmarks_count, _ = landmarks_batch.shape
        images = np.zeros([batch_size, sequence_length, *self.output_size, 3], dtype=np.uint8)
        for i in range(batch_size):
            for j in range(sequence_length):
                previous_position = None
                for k in range(landmarks_count):
                    x, y = landmarks_batch[i, j, k]
                    position = (int(x * self.output_size[0]), int(y * self.output_size[1]))
                    if previous_position is not None and k not in sections:
                        cv2.line(images[i, j], previous_position, position, color, thickness=self.line_thickness)
                    previous_position = position
        return images
