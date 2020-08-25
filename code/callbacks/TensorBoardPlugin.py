from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Union

from tensorflow.python.keras.callbacks import Callback, TensorBoard


class TensorBoardPlugin(Callback, ABC):
    def __init__(self,
                 tensorboard: TensorBoard,
                 update_freq: Union[str, int],
                 epoch_freq: int = None):
        super(TensorBoardPlugin, self).__init__()
        self.tensorboard: TensorBoard = tensorboard

        if update_freq == "batch":
            self.update_freq = 1
        else:
            self.update_freq = update_freq

        if update_freq == "epoch":
            self.epoch_freq = epoch_freq
        else:
            self.epoch_freq = None

        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != "epoch":
            self.samples_seen += logs["size"]
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen

    def on_epoch_end(self, epoch, logs=None):
        if self.update_freq == "epoch":
            index = epoch
        else:
            index = self.samples_seen

        if (self.epoch_freq is None) or (((epoch + 1) % self.epoch_freq) == 0):
            self._write_logs(index)

    def _get_writer(self, writer_name: str) -> tf.summary.SummaryWriter:
        # noinspection PyProtectedMember
        return self.tensorboard._get_writer(writer_name)

    @property
    def train_run_name(self) -> str:
        # noinspection PyProtectedMember
        return self.tensorboard._train_run_name

    @property
    def validation_run_name(self) -> str:
        # noinspection PyProtectedMember
        return self.tensorboard._validation_run_name

    @property
    def train_run_writer(self) -> tf.summary.SummaryWriter:
        return self._get_writer(self.train_run_name)

    @property
    def validation_run_writer(self) -> tf.summary.SummaryWriter:
        return self._get_writer(self.validation_run_name)

    @abstractmethod
    def _write_logs(self, index: int):
        raise NotImplementedError
