import tensorflow as tf
from datasets.data_augmentation import DataAugmentation


class VideoAugmentation(DataAugmentation):
    def __init__(self,
                 gain=None,
                 bias=None,
                 values_range=(0.0, 1.0),
                 apply_on_outputs=True):
        super(VideoAugmentation, self).__init__()
        self.gain = gain
        self.bias = bias
        self.values_range = values_range
        self.apply_on_outputs = apply_on_outputs

    def process(self, inputs: tf.data.Dataset, outputs: tf.data.Dataset):
        raise NotImplementedError
