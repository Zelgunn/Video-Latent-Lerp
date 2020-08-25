import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import def_function
from typing import List, Union

from callbacks import TensorBoardPlugin
from datasets import SubsetLoader
from modalities import Pattern


class AudioCallback(TensorBoardPlugin):
    def __init__(self,
                 summary_function: def_function.Function,
                 summary_inputs,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 is_one_shot=False):
        super(AudioCallback, self).__init__(tensorboard, update_freq, epoch_freq)
        self.summary_function = summary_function
        self.summary_inputs = summary_inputs
        self.writer_name = self.train_run_name if is_train_callback else self.validation_run_name
        self.is_one_shot = is_one_shot

    def _write_logs(self, index):
        if self.is_one_shot and self.summary_function is None:
            return

        with self._get_writer(self.writer_name).as_default():
            if self.summary_function is not None:
                self.summary_function(self.summary_inputs, step=index)

        if self.is_one_shot:
            self.summary_function = None

    @staticmethod
    def from_model_and_subset(autoencoder: keras.Model,
                              subset: Union[SubsetLoader],
                              pattern: Pattern,
                              name: str,
                              is_train_callback: bool,
                              tensorboard: keras.callbacks.TensorBoard,
                              update_freq="epoch",
                              epoch_freq=1,
                              sample_rate=48000,
                              ) -> List["AudioCallback"]:
        inputs, outputs = subset.get_batch(batch_size=4, pattern=pattern)

        # TODO : Do not use static values for mel spectrogram conversion
        def true_sound_function(data, step):
            return tf.summary.audio(name="{}_true".format(name), data=data, sample_rate=sample_rate,
                                    step=step, max_outputs=4)

        def pred_sound_function(data, step):
            predicted = autoencoder.predict_on_batch(data)
            return tf.summary.audio(name="{}_pred".format(name), data=predicted, sample_rate=sample_rate,
                                    step=step, max_outputs=4)

        one_shot_callback = AudioCallback(summary_function=true_sound_function, summary_inputs=outputs,
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=True)

        repeated_callback = AudioCallback(summary_function=pred_sound_function, summary_inputs=inputs,
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=False)

        return [one_shot_callback, repeated_callback]
