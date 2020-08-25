from tensorflow.python.keras .models import Model as KerasModel
from tensorflow.python.keras.callbacks import Callback
from typing import Dict
import os


class MultipleModelsCheckpoint(Callback):
    def __init__(self,
                 base_filepath: str,
                 models: Dict[str, KerasModel],
                 period: int):
        super(MultipleModelsCheckpoint, self).__init__()
        self.models = models
        self.period = period
        if "{model_name}" not in base_filepath:
            base_filepath = base_filepath + "_{model_name}"
        self.base_filepath = base_filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if ((epoch + 1) % self.period) == 0:
            for model_name, model in self.models.items():
                filepath = self.base_filepath.format(model_name=model_name, epoch=epoch + 1, **logs)
                filepath = os.path.normpath(filepath)
                model.save_weights(filepath, overwrite=True)
