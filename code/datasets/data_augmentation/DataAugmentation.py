import tensorflow as tf
from abc import ABC, abstractmethod


class DataAugmentation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, inputs: tf.data.Dataset, outputs: tf.data.Dataset):
        raise NotImplementedError
