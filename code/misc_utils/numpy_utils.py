import numpy as np


class NumpySeedContext(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        np.random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.seed(None)


def fast_concatenate_0(a_tuple):
    if len(a_tuple) == 1:
        return a_tuple[0]
    else:
        return np.concatenate(a_tuple, axis=0)
