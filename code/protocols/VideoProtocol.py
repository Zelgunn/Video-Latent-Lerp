import tensorflow as tf
from tensorflow.python.keras import Model
from abc import abstractmethod
from typing import List

from protocols import DatasetProtocol
from protocols import ImageCallbackConfig
from protocols.utils import make_residual_encoder, make_residual_decoder
from protocols.utils import video_random_cropping, video_dropout_noise, random_image_negative
from modalities import Pattern, ModalityLoadInfo, RawVideo
from models import AE, IAE


class VideoProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 initial_epoch: int,
                 model_name: str = None
                 ):
        super(VideoProtocol, self).__init__(dataset_name=dataset_name,
                                            protocol_name="video",
                                            initial_epoch=initial_epoch,
                                            model_name=model_name)

    # region Make model
    def make_model(self) -> Model:
        if self.model_architecture == "ae":
            return self.make_ae()
        elif self.model_architecture == "iae":
            return self.make_iae()
        else:
            raise ValueError("Unknown architecture : {}.".format(self.model_architecture))

    def make_ae(self) -> AE:
        encoder = self.make_encoder(self.get_encoder_input_shape()[1:])
        decoder = self.make_decoder(self.get_latent_code_shape(encoder)[1:])

        model = AE(encoder=encoder,
                   decoder=decoder,
                   learning_rate=WarmupSchedule(1000, self.learning_rate))
        return model

    def make_iae(self) -> IAE:
        encoder = self.make_encoder(self.get_encoder_input_shape()[1:])
        decoder = self.make_decoder(self.get_latent_code_shape(encoder)[1:])

        model = IAE(encoder=encoder,
                    decoder=decoder,
                    step_size=self.step_size,
                    learning_rate=WarmupSchedule(1000, self.learning_rate))
        return model

    # endregion

    # region Sub-models input shapes
    def get_encoder_input_shape(self):
        shape = (None, self.step_size, self.height, self.width, 1)
        return shape

    def get_latent_code_shape(self, encoder: Model):
        shape = encoder.compute_output_shape(self.get_encoder_input_shape())
        if self.model_architecture == "vaegan":
            shape = (*shape[:-1], shape[-1] // 2)
        return shape

    # endregion

    # region Make sub-models
    def make_encoder(self, input_shape) -> Model:
        encoder = make_residual_encoder(input_shape=input_shape,
                                        filters=self.encoder_filters,
                                        kernel_size=self.kernel_size,
                                        strides=self.encoder_strides,
                                        code_size=self.code_size,
                                        code_activation=self.code_activation,
                                        use_batch_norm=self.use_batch_norm,
                                        use_residual_bias=self.encoder_config["use_residual_bias"],
                                        use_conv_bias=self.encoder_config["use_conv_bias"],
                                        )
        return encoder

    def make_decoder(self, input_shape) -> Model:
        decoder = make_residual_decoder(input_shape=input_shape,
                                        filters=self.decoder_filters,
                                        kernel_size=self.kernel_size,
                                        strides=self.decoder_strides,
                                        channels=1,
                                        output_activation=self.output_activation,
                                        use_batch_norm=self.use_batch_norm,
                                        use_residual_bias=self.decoder_config["use_residual_bias"],
                                        use_conv_bias=self.decoder_config["use_conv_bias"],
                                        )
        return decoder

    # endregion

    # region Patterns
    def get_train_pattern(self) -> Pattern:
        augment_video = self.make_video_augmentation()

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            output_map=augment_video
        )
        return pattern

    def get_image_pattern(self) -> Pattern:
        video_preprocess = self.make_video_preprocess()

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            output_map=video_preprocess
        )
        return pattern

    def get_anomaly_pattern(self) -> Pattern:
        return self.get_image_pattern().with_labels()

    # endregion

    # region Pre-processes
    def make_video_augmentation(self):
        preprocess_video = self.make_video_preprocess()

        def augment_video(video: tf.Tensor) -> tf.Tensor:
            if self.use_cropping:
                video = video_random_cropping(video, self.crop_ratio, self.output_length)

            if self.dropout_noise_ratio > 0.0:
                video = video_dropout_noise(video, self.dropout_noise_ratio, spatial_prob=0.1)

            if self.use_random_negative:
                video = random_image_negative(video, negative_prob=0.5)

            video = preprocess_video(video)
            return video

        return augment_video

    def make_video_preprocess(self):
        def preprocess(video: tf.Tensor, labels: tf.Tensor = None):
            video = tf.image.resize(video, (self.height, self.width))

            if self.dataset_channels == 3:
                rgb_weights = [0.2989, 0.5870, 0.1140]
                rgb_weights = tf.reshape(rgb_weights, [1, 1, 1, 3])
                video *= rgb_weights
                video = tf.reduce_sum(video, axis=-1, keepdims=True)

            if labels is None:
                return video
            else:
                return video, labels

        return preprocess

    # endregion

    def get_image_callback_configs(self):
        image_pattern = self.get_image_pattern()

        image_callbacks_configs = [
            ImageCallbackConfig(self.model, image_pattern, True, "train"),
            ImageCallbackConfig(self.model, image_pattern, False, "test"),
        ]

        if isinstance(self.model, IAE):
            image_callbacks_configs += \
                [ImageCallbackConfig(self.model.interpolate, image_pattern, False, "interpolate_test")]

        return image_callbacks_configs

    # region Properties
    # region Abstract properties
    @property
    @abstractmethod
    def dataset_channels(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def use_face(self) -> bool:
        raise NotImplementedError

    # endregion
    # region Config properties
    @property
    def model_architecture(self) -> str:
        return self.config["model_architecture"].lower()

    @property
    def height(self) -> int:
        return self.config["height"]

    @property
    def width(self) -> int:
        return self.config["width"]

    @property
    def step_size(self) -> int:
        return self.config["step_size"]

    @property
    def step_count(self) -> int:
        return self.config["step_count"]

    @property
    def code_size(self) -> int:
        return self.config["code_size"]

    # region Encoder
    @property
    def encoder_config(self):
        return self.config["encoder"]

    @property
    def encoder_filters(self) -> List[int]:
        return self.encoder_config["filters"]

    @property
    def encoder_strides(self) -> List[List[int]]:
        return self.encoder_config["strides"]

    @property
    def code_activation(self) -> str:
        return self.config["code_activation"]

    # endregion

    # region Decoder
    @property
    def decoder_config(self):
        return self.config["decoder"]

    @property
    def decoder_filters(self) -> List[int]:
        return self.decoder_config["filters"]

    @property
    def decoder_strides(self) -> List[List[int]]:
        return self.decoder_config["strides"]

    @property
    def output_activation(self) -> str:
        return self.config["output_activation"]

    # endregion

    @property
    def kernel_size(self) -> int:
        return self.config["kernel_size"]

    @property
    def use_batch_norm(self) -> bool:
        return self.config["use_batch_norm"]

    @property
    def batch_size(self) -> int:
        return self.config["batch_size"]

    @property
    def learning_rate(self) -> float:
        return self.config["learning_rate"]

    # region Data augmentation
    @property
    def data_augmentation_config(self):
        return self.config["data_augmentation"]

    @property
    def use_cropping(self) -> bool:
        return self.crop_ratio > 0.0

    @property
    def crop_ratio(self) -> float:
        return self.data_augmentation_config["crop_ratio"]

    @property
    def dropout_noise_ratio(self) -> float:
        return self.data_augmentation_config["dropout_noise_ratio"]

    @property
    def gaussian_noise_ratio(self) -> float:
        return self.data_augmentation_config["gaussian_noise_ratio"]

    @property
    def use_random_negative(self) -> bool:
        return self.data_augmentation_config["use_random_negative"]

    # endregion

    # endregion

    @property
    def output_length(self) -> int:
        if self.model_architecture in ["iae", "and", "iaegan"]:
            return self.step_size * self.step_count
        else:
            return self.step_size

    # endregion

    def autoencode_video(self, video_source, target_path: str, load_epoch: int, fps=25.0):
        from datasets.data_readers import VideoReader
        from tqdm import tqdm
        import numpy as np
        import cv2

        self.load_weights(epoch=load_epoch)

        video_reader = VideoReader(video_source)
        output_size = (self.width, self.height)
        video_writer = cv2.VideoWriter(target_path, cv2.VideoWriter.fourcc(*"H264"),
                                       fps, output_size)
        frames = []

        for frame in tqdm(video_reader, total=video_reader.frame_count):
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32)
                frame /= 255
            frame = cv2.resize(frame, dsize=(self.width, self.height))
            if frame.ndim == 2:
                frame = np.expand_dims(frame, axis=-1)
            else:
                frame = np.mean(frame, axis=-1, keepdims=True)
            frames.append(frame)

            if len(frames) == self.output_length:
                inputs = np.expand_dims(np.stack(frames, axis=0), axis=0)
                outputs = self.autoencoder(inputs)
                for output in outputs.numpy()[0]:
                    output = cv2.resize(output, output_size)
                    output = np.clip(output, 0.0, 1.0)
                    output = (output * 255).astype(np.uint8)
                    if output.ndim == 2:
                        output = np.expand_dims(output, axis=-1)
                        output = np.tile(output, reps=[1, 1, 3])
                    video_writer.write(output)
                frames = []

        video_writer.release()


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps: int, learning_rate=1e-3):
        super(WarmupSchedule, self).__init__()

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        factor = (step + 1) / self.warmup_steps
        return self.learning_rate * tf.math.minimum(factor, 1.0)

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        return config
