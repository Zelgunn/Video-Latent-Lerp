import tensorflow as tf


class EagerModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath: str, verbose=0):
        super(EagerModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        target_path = self.filepath.format(epoch=epoch + 1)
        if self.verbose > 0:
            print("\nEpoch {epoch:05d}: saving model to {path}".format(epoch=epoch + 1, path=target_path))
        self.model.save(target_path,
                        include_optimizer=True)
        exit()
