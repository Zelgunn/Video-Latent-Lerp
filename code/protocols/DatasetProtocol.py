from tensorflow.python.keras import Model
from abc import abstractmethod
from typing import Callable, Dict, Optional, List
import json
import os

from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from protocols import AUCCallbackConfig
from modalities import Pattern
from models import IAE


class DatasetProtocol(Protocol):
    def __init__(self,
                 dataset_name: str,
                 protocol_name: str,
                 initial_epoch: int,
                 model_name: str = None
                 ):

        self.config = self.load_config(protocol_name, dataset_name)
        model = self.make_model()
        autoencoder = self.make_autoencoder(model)
        self.initial_epoch = initial_epoch

        super(DatasetProtocol, self).__init__(model=model,
                                              dataset_name=dataset_name,
                                              protocol_name=protocol_name,
                                              autoencoder=autoencoder,
                                              model_name=model_name)

    # region Init
    @abstractmethod
    def make_model(self) -> Model:
        raise NotImplementedError

    @staticmethod
    def make_autoencoder(model: Model) -> Optional[Callable]:
        return model

    # endregion

    # region Train
    def train_model(self, config: ProtocolTrainConfig = None):
        if config is None:
            config = self.get_train_config()

        super(DatasetProtocol, self).train_model(config)

    @abstractmethod
    def get_train_config(self) -> ProtocolTrainConfig:
        raise NotImplementedError

    # endregion

    # region Test
    def test_model(self, config: ProtocolTestConfig = None):
        if config is None:
            config = self.get_test_config()

        super(DatasetProtocol, self).test_model(config)

    @abstractmethod
    def get_test_config(self) -> ProtocolTestConfig:
        raise NotImplementedError

    # endregion

    # region Patterns
    @abstractmethod
    def get_train_pattern(self) -> Pattern:
        raise NotImplementedError

    @abstractmethod
    def get_anomaly_pattern(self) -> Pattern:
        raise NotImplementedError

    # endregion

    def get_auc_callbacks_configs(self) -> List[AUCCallbackConfig]:
        anomaly_pattern = self.get_anomaly_pattern()

        auc_callbacks_configs = [
            AUCCallbackConfig(self.model, anomaly_pattern, self.output_length, prefix=""),

        ]

        if isinstance(self.model, IAE):
            auc_callbacks_configs += \
                [AUCCallbackConfig(self.model.interpolate, anomaly_pattern, self.output_length, prefix="iae")]

        return auc_callbacks_configs

    def make_log_dir(self, sub_folder: str) -> str:
        log_dir = super(DatasetProtocol, self).make_log_dir(sub_folder)
        self.save_config(log_dir)
        return log_dir

    # region Config
    @property
    @abstractmethod
    def output_length(self) -> int:
        raise NotImplementedError

    def get_config_path(self, protocol_name: str = None, dataset_name: str = None):
        if protocol_name is None:
            protocol_name = self.protocol_name
        if dataset_name is None:
            protocol_name = self.dataset_name
        return "protocols/configs/{protocol_name}/{dataset_name}.json" \
            .format(protocol_name=protocol_name, dataset_name=dataset_name)

    def load_config(self, protocol_name: str = None, dataset_name: str = None) -> Dict:
        config_path = self.get_config_path(protocol_name, dataset_name)
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config

    def save_config(self, log_dir: str):
        config_path = os.path.join(log_dir, "main_config.json")
        with open(config_path, 'w') as config_file:
            json.dump(self.config, config_file)
    # endregion
