# import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import cv2
from typing import Tuple, List


def plot_line2d_to_array(x: np.ndarray, y: np.ndarray,
                         output_size: Tuple or List = None, dpi=75.0,
                         normalize=True, use_xy_limits=True) -> np.ndarray:
    if normalize:
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

    if output_size is None:
        figsize = plt.rcParams["figure.figsize"]
    else:
        figsize = [output_size[0] / dpi, output_size[1] / dpi]

    figure = plt.Figure(figsize=figsize, dpi=dpi)
    plot = figure.add_subplot(111)
    if use_xy_limits:
        plot.set_xlim(0.0, 1.0)
        plot.set_ylim(0.0, 1.0)
    plot.plot(x, y)

    canvas = FigureCanvasAgg(figure)
    canvas.draw()

    # noinspection PyTypeChecker
    canvas_as_str: str = canvas.tostring_rgb()
    image = np.fromstring(canvas_as_str, dtype="uint8")
    image = np.reshape(image, [output_size[0], output_size[1], 3])

    return image


def resized_one(image, size) -> np.ndarray:
    result = cv2.resize(image, tuple(reversed(size)))
    if image.shape[-1] == 1:
        return np.expand_dims(result, -1)
    else:
        return result
