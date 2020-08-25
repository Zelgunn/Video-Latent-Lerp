import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import def_function
from typing import List, Union, Callable

from callbacks import TensorBoardPlugin
from misc_utils.summary_utils import image_summary
from datasets import SubsetLoader
from modalities import Pattern


class ImageCallback(TensorBoardPlugin):
    def __init__(self,
                 summary_function: def_function.Function,
                 summary_inputs,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 is_one_shot=False):
        super(ImageCallback, self).__init__(tensorboard, update_freq, epoch_freq)
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
    def from_model_and_subset(autoencoder: Callable,
                              subset: Union[SubsetLoader],
                              pattern: Pattern,
                              name: str,
                              is_train_callback: bool,
                              tensorboard: keras.callbacks.TensorBoard,
                              update_freq="epoch",
                              epoch_freq=1,
                              max_outputs=4,
                              ) -> List["ImageCallback"]:
        batch = subset.get_batch(batch_size=4, pattern=pattern)
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            inputs, outputs = batch
        else:
            inputs = outputs = batch

        if len(inputs.shape) >= 5:
            one_shot_base_function = video_summary
            repeated_base_function = video_autoencoder_summary
        else:
            one_shot_base_function = image_summary
            repeated_base_function = image_autoencoder_summary

        def one_shot_function(data, step):
            data = convert_tensor_uint8(data)
            return one_shot_base_function(name=name, data=data, step=step, max_outputs=max_outputs)

        def repeated_function(data, step):
            _inputs, _outputs = data
            decoded = autoencoder(_inputs)
            return repeated_base_function(name=name, true_data=_outputs, pred_data=decoded, step=step)

        one_shot_callback = ImageCallback(summary_function=one_shot_function, summary_inputs=inputs,
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=True)

        repeated_callback = ImageCallback(summary_function=repeated_function, summary_inputs=(inputs, outputs),
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=False)

        return [one_shot_callback, repeated_callback]


# region Utility / wrappers
def video_summary(name: str,
                  data: tf.Tensor,
                  fps=(8, 25),
                  step: int = None,
                  max_outputs=4
                  ):
    if data.dtype != tf.uint8:
        data = convert_tensor_uint8(data)
    for _fps in fps:
        image_summary(name="{}_{}".format(name, _fps),
                      data=data,
                      fps=_fps,
                      step=step,
                      max_outputs=max_outputs,
                      )


def video_autoencoder_summary(name: str,
                              true_data: tf.Tensor,
                              pred_data: tf.Tensor,
                              fps=(8, 25),
                              step: int = None,
                              max_outputs=4):
    true_data = convert_tensor_uint8(true_data)
    pred_data = convert_tensor_uint8(pred_data)

    for _fps in fps:
        image_summary(name="{}_pred_outputs_{}".format(name, _fps), data=pred_data,
                      step=step, max_outputs=max_outputs, fps=_fps)

    if pred_data.shape.is_compatible_with(true_data.shape):
        delta = (pred_data - true_data) * (tf.cast(pred_data < true_data, dtype=tf.uint8) * 254 + 1)
        for _fps in fps:
            image_summary(name="{}_delta_{}".format(name, _fps), data=delta,
                          step=step, max_outputs=max_outputs, fps=_fps)


def image_autoencoder_summary(name: str,
                              true_data: tf.Tensor,
                              pred_data: tf.Tensor,
                              step: int = None,
                              max_outputs=4):
    true_data = convert_tensor_uint8(true_data)
    pred_data = convert_tensor_uint8(pred_data)

    image_summary(name="{}_pred_outputs".format(name), data=pred_data, step=step, max_outputs=max_outputs)

    if pred_data.shape.is_compatible_with(true_data.shape):
        delta = (pred_data - true_data) * (tf.cast(pred_data < true_data, dtype=tf.uint8) * 254 + 1)
        image_summary(name="{}_delta".format(name), data=delta, step=step, max_outputs=max_outputs)


def convert_tensor_uint8(tensor) -> tf.Tensor:
    tensor: tf.Tensor = tf.convert_to_tensor(tensor)
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized = tf.cast(tensor * tf.constant(255, dtype=tensor.dtype), tf.uint8)
    return normalized
# endregion
