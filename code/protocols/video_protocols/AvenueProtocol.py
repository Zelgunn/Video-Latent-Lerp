from protocols import VideoProtocol, ProtocolTrainConfig, ProtocolTestConfig


class AvenueProtocol(VideoProtocol):
    def __init__(self,
                 initial_epoch=0,
                 model_name=None
                 ):
        super(AvenueProtocol, self).__init__(dataset_name="avenue",
                                             initial_epoch=initial_epoch,
                                             model_name=model_name)

    def get_train_config(self) -> ProtocolTrainConfig:
        train_pattern = self.get_train_pattern()
        image_callbacks_configs = self.get_image_callback_configs()
        auc_callbacks_configs = self.get_auc_callbacks_configs()

        return ProtocolTrainConfig(batch_size=self.batch_size,
                                   pattern=train_pattern,
                                   epochs=50,
                                   initial_epoch=self.initial_epoch,
                                   image_callbacks_configs=image_callbacks_configs,
                                   auc_callbacks_configs=auc_callbacks_configs,
                                   early_stopping_metric=self.model.metrics_names[0])

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()
        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.initial_epoch,
                                  output_length=self.output_length,
                                  detector_stride=1,
                                  pre_normalize_predictions=True)

    @property
    def dataset_channels(self) -> int:
        return 3

    @property
    def use_face(self) -> bool:
        return False
