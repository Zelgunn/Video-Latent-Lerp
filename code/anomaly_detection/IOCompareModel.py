from tensorflow.python.keras import Model
from typing import Union, List, Callable

from anomaly_detection import IOCompareLayer


class IOCompareModel(Model):
    def __init__(self,
                 autoencoder: Callable,
                 output_length: int,
                 metrics: List[Union[str, Callable]] = "mse",
                 **kwargs):
        super(IOCompareModel, self).__init__(**kwargs)
        self.autoencoder = autoencoder
        self.output_length = output_length

        if not isinstance(metrics, list):
            metrics = [metrics]

        self.io_compare_layers = []

        for metric in metrics:
            layer = IOCompareLayer(metric=metric, output_length=output_length)
            self.io_compare_layers.append(layer)

    def call(self, inputs, training=None, mask=None):
        inputs, ground_truth = inputs
        decoded = self.autoencoder(inputs)

        predictions = []
        for io_compare_layer in self.io_compare_layers:
            layer_predictions = io_compare_layer([decoded, ground_truth])
            predictions.append(layer_predictions)

        return predictions

    def compute_output_signature(self, input_signature):
        raise NotImplementedError
