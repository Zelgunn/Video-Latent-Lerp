import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from models import CustomModel


class AE(CustomModel):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate=1e-3,
                 **kwargs):
        super(AE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate

        self.optimizer = None
        self.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def call(self, inputs, training=None, mask=None):
        return self.decode(self.encode(inputs))

    @tf.function
    def encode(self, inputs):
        return self.encoder(inputs)

    @tf.function
    def decode(self, inputs):
        return self.decoder(inputs)

    @property
    def metrics_names(self):
        return ["reconstruction"]

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> tf.Tensor:
        outputs = self(inputs)
        return tf.reduce_mean(tf.square(inputs - outputs))

    def compute_encoded_shape(self, input_shape):
        return self.encoder.compute_output_shape(input_shape)

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "learning_rate": self.learning_rate
        }
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {self.encoder: "encoder",
                self.decoder: "decoder"}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.encoder.optimizer = optimizer
        self.decoder.optimizer = optimizer
