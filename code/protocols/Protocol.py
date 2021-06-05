import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
import time
import os
from typing import List, Callable

from anomaly_detection import AnomalyDetector, known_metrics, IOCompareModel
from datasets import DatasetLoader, DatasetConfig
from misc_utils.train_utils import save_model_info
from callbacks import ImageCallback, AUCCallback
from modalities import Pattern


class ImageCallbackConfig(object):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 is_train_callback: bool,
                 name: str,
                 epoch_freq: int = 1
                 ):
        self.autoencoder = autoencoder
        self.pattern = pattern
        self.name = name
        self.is_train_callback = is_train_callback
        self.epoch_freq = epoch_freq

    def to_callbacks(self,
                     tensorboard: TensorBoard,
                     dataset_loader: DatasetLoader,
                     ) -> List[ImageCallback]:
        subset = dataset_loader.train_subset if self.is_train_callback else dataset_loader.test_subset
        image_callbacks = ImageCallback.from_model_and_subset(autoencoder=self.autoencoder,
                                                              subset=subset,
                                                              pattern=self.pattern,
                                                              name=self.name,
                                                              is_train_callback=self.is_train_callback,
                                                              tensorboard=tensorboard,
                                                              epoch_freq=self.epoch_freq)
        return image_callbacks


class AUCCallbackConfig(object):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 output_length: int,
                 prefix: str,
                 epoch_freq: int = 1
                 ):
        self.autoencoder = autoencoder
        self.pattern = pattern
        self.output_length = output_length
        self.prefix = prefix
        self.epoch_freq = epoch_freq

    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    ) -> AUCCallback:
        raw_predictions_model = IOCompareModel(self.autoencoder,
                                               output_length=self.output_length,
                                               name="{}AutoencoderRawPredictionsModel".format(self.prefix))

        return AUCCallback.from_subset(predictions_model=raw_predictions_model,
                                       tensorboard=tensorboard,
                                       test_subset=dataset_loader.test_subset,
                                       pattern=self.pattern,
                                       samples_count=512,
                                       epoch_freq=self.epoch_freq,
                                       batch_size=4,
                                       prefix=self.prefix)


class ProtocolTrainConfig(object):
    def __init__(self,
                 batch_size: int,
                 pattern: Pattern,
                 epochs: int,
                 initial_epoch: int,
                 image_callbacks_configs: List[ImageCallbackConfig] = None,
                 auc_callbacks_configs: List[AUCCallbackConfig] = None,
                 early_stopping_metric: str = None,
                 ):
        self.batch_size = batch_size
        self.pattern = pattern
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.image_callbacks_configs = image_callbacks_configs
        self.auc_callbacks_configs = auc_callbacks_configs
        self.early_stopping_metric = early_stopping_metric


class ProtocolTestConfig(object):
    def __init__(self,
                 pattern: Pattern,
                 epoch: int,
                 output_length: int,
                 detector_stride: int,
                 pre_normalize_predictions: bool,
                 additional_metrics: List[Callable[[tf.Tensor], tf.Tensor]] = None,
                 **kwargs,
                 ):
        self.pattern = pattern
        self.epoch = epoch
        self.output_length = output_length
        self.detector_stride = detector_stride
        self.pre_normalize_predictions = pre_normalize_predictions
        self.additional_metrics = additional_metrics
        self.kwargs = kwargs


class Protocol(object):
    def __init__(self,
                 model: Model,
                 dataset_name: str,
                 protocol_name: str,
                 autoencoder: Callable = None,
                 model_name: str = None
                 ):
        self.model = model
        if autoencoder is None:
            autoencoder = model
        self.autoencoder = autoencoder
        if model_name is None:
            model_name = model.name
        self.model_name = model_name
        self.autoencoder_name = autoencoder.name if hasattr(autoencoder, "name") else autoencoder.__name__
        self.protocol_name = protocol_name

        self.dataset_name = dataset_name
        self.dataset_folder = get_dataset_folder(dataset_name)
        self.dataset_config = DatasetConfig(self.dataset_folder, output_range=(0.0, 1.0))
        self.dataset_loader = DatasetLoader(self.dataset_config)

        self.base_log_dir = "../logs/{dataset_name}" \
            .format(protocol_name=protocol_name, dataset_name=dataset_name)

    def train_model(self, config: ProtocolTrainConfig):
        self.load_weights(epoch=config.initial_epoch)

        log_dir = self.make_log_dir("train")
        callbacks = self.make_callback(log_dir, config)

        subset = self.dataset_loader.train_subset
        train_dataset, val_dataset = subset.make_tf_datasets_splits(config.pattern, split=0.8)
        train_dataset = train_dataset.batch(config.batch_size).prefetch(-1)
        if val_dataset is not None:
            val_dataset = val_dataset.batch(config.batch_size)

        self.model.fit(train_dataset, steps_per_epoch=1000, epochs=config.epochs,
                       validation_data=val_dataset, validation_steps=128,
                       callbacks=callbacks, initial_epoch=config.initial_epoch)

    def make_callback(self,
                      log_dir: str,
                      config: ProtocolTrainConfig
                      ) -> List[Callback]:
        tensorboard = TensorBoard(log_dir=log_dir, update_freq=16, profile_batch=0)
        callbacks = [tensorboard]
        # region Image Callbacks
        if config.image_callbacks_configs is not None:
            for icc in config.image_callbacks_configs:
                callbacks += icc.to_callbacks(tensorboard, self.dataset_loader)
        # endregion
        # region Checkpoint
        weights_path = os.path.join(log_dir, "weights_{epoch:03d}.hdf5")
        model_checkpoint = ModelCheckpoint(weights_path)
        callbacks.append(model_checkpoint)
        # endregion
        # region Early stopping
        if config.early_stopping_metric is not None:
            early_stopping = EarlyStopping(monitor=config.early_stopping_metric,
                                           mode="min",
                                           patience=5
                                           )
            callbacks.append(early_stopping)
        # endregion
        # region AUC
        if config.auc_callbacks_configs is not None:
            for acc in config.auc_callbacks_configs:
                callbacks += [acc.to_callback(tensorboard, self.dataset_loader)]
        # endregion
        return callbacks

    def test_model(self, config: ProtocolTestConfig):
        self.load_weights(epoch=config.epoch)

        compare_metrics = list(known_metrics.keys())
        additional_metrics = self.additional_test_metrics
        if config.additional_metrics is not None:
            additional_metrics = [*additional_metrics, *config.additional_metrics]

        anomaly_detector = AnomalyDetector(autoencoder=self.autoencoder,
                                           output_length=config.output_length,
                                           compare_metrics=compare_metrics,
                                           additional_metrics=additional_metrics)

        log_dir = self.make_log_dir("anomaly_detection")
        pattern = config.pattern  # .with_added_depth().with_added_depth()

        if self.dataset_name is "emoly":
            folders = self.dataset_loader.test_subset.subset_folders
            folders = [folder for folder in folders if "acted" in folder]
            self.dataset_loader.test_subset.subset_folders = folders

        anomaly_detector.predict_and_evaluate(dataset=self.dataset_loader,
                                              pattern=pattern,
                                              log_dir=log_dir,
                                              stride=config.detector_stride,
                                              pre_normalize_predictions=config.pre_normalize_predictions,
                                              additional_config={
                                                  "epoch": config.epoch,
                                                  "model_name": self.autoencoder_name,
                                                  **config.kwargs
                                              }
                                              )

    @property
    def additional_test_metrics(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return getattr(self.model, "additional_test_metrics", [])

    def multi_test_model(self, configs: List[ProtocolTestConfig]):
        for config in configs:
            self.test_model(config)

    def make_log_dir(self, sub_folder: str) -> str:
        timestamp = int(time.time())
        log_dir = os.path.join(self.base_log_dir, sub_folder, "{}_{}".format(timestamp, self.model_name))
        os.makedirs(log_dir)
        save_model_info(self.model, log_dir)
        return log_dir

    def load_weights(self, epoch: int):
        if epoch > 0:
            weights_path = os.path.join(self.base_log_dir, "weights_{epoch:03d}.hdf5")
            self.model.load_weights(weights_path.format(epoch=epoch))


def get_dataset_folder(dataset_name: str) -> str:
    known_datasets = {
        "ped2": "../datasets/ucsd/ped2",
        "ped1": "../datasets/ucsd/ped1",

        "subway_exit": "../datasets/subway/exit",
        "subway_entrance": "../datasets/subway/entrance",
        "subway_mall1": "../datasets/subway/mall3",
        "subway_mall2": "../datasets/subway/mall3",
        "subway_mall3": "../datasets/subway/mall3",

        "shanghaitech": "../datasets/shanghaitech",
        "emoly": "../datasets/emoly",
        "avenue": "../datasets/avenue",
    }

    if dataset_name in known_datasets:
        return known_datasets[dataset_name]
    else:
        raise ValueError("Unknown dataset : {}".format(dataset_name))
