import tensorflow as tf
from tensorflow.python.keras.models import Model
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from typing import Union, List, Callable, Dict, Any, Tuple

from anomaly_detection import IOCompareModel
from datasets import DatasetLoader, SubsetLoader
from modalities import Pattern


class AnomalyDetector(Model):
    def __init__(self,
                 autoencoder: Callable,
                 output_length: int,
                 compare_metrics: List[Union[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]] = "mse",
                 additional_metrics: List[Callable[[tf.Tensor], tf.Tensor]] = None,
                 **kwargs
                 ):
        """

        :param autoencoder:
        :param output_length:
        :param compare_metrics:
        :param additional_metrics:
        :param kwargs:
        """
        super(AnomalyDetector, self).__init__(**kwargs)

        self.io_compare_model = IOCompareModel(autoencoder=autoencoder,
                                               output_length=output_length,
                                               metrics=compare_metrics)
        self.additional_metrics = to_list(additional_metrics) if additional_metrics is not None else []

        compare_metrics = to_list(compare_metrics)

        self.anomaly_metrics_names = []
        all_metrics = compare_metrics + self.additional_metrics
        for metric in all_metrics:
            metric_name = metric if isinstance(metric, str) else metric.__name__
            self.anomaly_metrics_names.append(metric_name)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs, ground_truth = inputs
        predictions = self.io_compare_model([inputs, ground_truth])

        for additional_metric in self.additional_metrics:
            predictions.append(additional_metric(inputs))

        return predictions

    def predict_and_evaluate(self,
                             dataset: DatasetLoader,
                             pattern: Pattern,
                             log_dir: str,
                             stride=1,
                             pre_normalize_predictions=True,
                             max_samples=-1,
                             additional_config: Dict[str, Any] = None,
                             ):
        predictions, labels = self.predict_anomalies(dataset=dataset,
                                                     pattern=pattern,
                                                     stride=stride,
                                                     pre_normalize_predictions=pre_normalize_predictions,
                                                     max_samples=max_samples)

        merged_predictions, merged_labels = self.merge_samples_predictions(predictions=predictions, labels=labels)
        self.save_predictions(predictions=merged_predictions, labels=labels, log_dir=log_dir)
        results = self.evaluate_predictions(predictions=merged_predictions, labels=merged_labels)

        samples_names = [os.path.basename(folder) for folder in dataset.test_subset.subset_folders]
        self.plot_predictions(predictions=predictions,
                              labels=labels,
                              log_dir=log_dir,
                              samples_names=samples_names)

        for i in range(self.metric_count):
            metric_results_string = "Anomaly_score ({}):".format(self.anomaly_metrics_names[i])
            for result_name, result_values in results.items():
                metric_results_string += " {} = {} |".format(result_name, result_values[i])
            print(metric_results_string)

        additional_config["stride"] = stride
        additional_config["pre-normalize predictions"] = pre_normalize_predictions
        self.save_evaluation_results(log_dir=log_dir,
                                     results=results,
                                     additional_config=additional_config)
        return results

    # region Predict anomaly scores
    def predict_anomalies(self,
                          dataset: DatasetLoader,
                          pattern: Pattern,
                          stride=1,
                          pre_normalize_predictions=True,
                          max_samples=10
                          ) -> Tuple[List[np.ndarray], np.ndarray]:
        predictions, labels = self.predict_anomalies_on_subset(subset=dataset.test_subset,
                                                               pattern=pattern,
                                                               stride=stride,
                                                               pre_normalize_predictions=pre_normalize_predictions,
                                                               max_samples=max_samples)

        return predictions, labels

    def predict_anomalies_on_subset(self,
                                    subset: SubsetLoader,
                                    pattern: Pattern,
                                    stride: int,
                                    pre_normalize_predictions: bool,
                                    max_samples=10
                                    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        labels = []
        predictions = [[] for _ in range(self.metric_count)]

        sample_count = min(max_samples, len(subset.subset_folders)) if max_samples > 0 else len(subset.subset_folders)
        print("Making predictions for {} videos".format(sample_count))

        for sample_index in range(sample_count):
            sample_name = subset.subset_folders[sample_index]
            print("Predicting on sample n{}/{} ({})".format(sample_index + 1, sample_count, sample_name))
            sample_results = self.predict_anomalies_on_sample(subset, pattern,
                                                              sample_index, stride,
                                                              normalize_predictions=pre_normalize_predictions)
            sample_predictions, sample_labels = sample_results
            for i in range(self.metric_count):
                predictions[i].append(sample_predictions[i])
            labels.append(sample_labels)

        return predictions, labels

    def predict_anomalies_on_sample(self,
                                    subset: SubsetLoader,
                                    pattern: Pattern,
                                    sample_index: int,
                                    stride: int,
                                    normalize_predictions=False,
                                    ):
        dataset = subset.make_source_browser(pattern, sample_index, stride)

        # modality_folder = os.path.join(subset.subset_folders[sample_index], "labels")
        # modality_files = [os.path.join(modality_folder, file)
        #                   for file in os.listdir(modality_folder) if file.endswith(".tfrecord")]
        # file_count = len(modality_files)
        # k = 0

        get_outputs_from_inputs = len(dataset.element_spec) == 2

        predictions, labels = None, []
        for sample in dataset:
            if get_outputs_from_inputs:
                sample_inputs, sample_labels = sample
                sample_outputs = sample_inputs
            else:
                sample_inputs, sample_outputs, sample_labels = sample

            sample_predictions = self([sample_inputs, sample_outputs])

            labels.append(sample_labels)
            if predictions is None:
                predictions = [[metric_prediction] for metric_prediction in sample_predictions]
            else:
                for i in range(len(predictions)):
                    predictions[i].append(sample_predictions[i])

            # print("{}/{}".format(k, file_count))
            # k += 1

        predictions = [np.concatenate(metric_prediction, axis=0) for metric_prediction in predictions]
        labels = np.concatenate(labels, axis=0)

        from datasets.loaders import SubsetLoader
        labels = SubsetLoader.timestamps_labels_to_frame_labels(labels, 32)
        mask = np.zeros_like(labels)
        mask[:, 15] = 1
        # mask[:, 16] = 1
        labels = np.sum(labels * mask, axis=-1) >= 1

        if normalize_predictions:
            for i in range(self.metric_count):
                metric_pred = predictions[i]
                metric_pred = (metric_pred - metric_pred.min()) / (metric_pred.max() - metric_pred.min())
                predictions[i] = metric_pred

        return predictions, labels

    # endregion

    def save_predictions(self, predictions: List[np.ndarray], labels: np.ndarray, log_dir: str):
        for i in range(self.metric_count):
            if predictions[i].ndim > 1:
                predictions[i] = np.mean(predictions[i], axis=tuple(range(1, predictions[i].ndim)))

        np.save(os.path.join(log_dir, "predictions.npy"), predictions)
        np.save(os.path.join(log_dir, "labels.npy"), labels)

    @staticmethod
    def merge_samples_predictions(predictions: List[List[np.ndarray]],
                                  labels: np.ndarray
                                  ) -> Tuple[List[np.ndarray], np.ndarray]:
        merged_predictions = []
        metric_count = len(predictions)
        for i in range(metric_count):
            metric_pred = np.concatenate(predictions[i])
            merged_predictions.append(metric_pred)

        labels = np.concatenate(labels)
        return merged_predictions, labels

    # region Evaluate predictions
    @staticmethod
    def evaluate_predictions(predictions: List[np.ndarray],
                             labels: np.ndarray
                             ) -> Dict[str, List[float]]:
        predictions = [(metric_pred - metric_pred.min()) / (metric_pred.max() - metric_pred.min())
                       for metric_pred in predictions]

        results = None
        for i in range(len(predictions)):
            metric_results = AnomalyDetector.evaluate_metric_predictions(predictions[i], labels)

            if results is None:
                results = {result_name: [] for result_name in metric_results}

            for result_name in metric_results:
                results[result_name].append(metric_results[result_name])

        return results

    @staticmethod
    def evaluate_metric_predictions(predictions: np.ndarray,
                                    labels: np.ndarray
                                    ):
        roc = tf.metrics.AUC(curve="ROC", num_thresholds=1000)
        pr = tf.metrics.AUC(curve="PR", num_thresholds=1000)

        thresholds = list(np.arange(0.01, 1.0, 1.0 / 200.0, dtype=np.float32))
        precision = tf.metrics.Precision(thresholds=thresholds)
        recall = tf.metrics.Recall(thresholds=thresholds)

        if predictions.ndim > 1 and labels.ndim == 1:
            predictions = predictions.mean(axis=tuple(range(1, predictions.ndim)))

        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        roc.update_state(labels, predictions)
        pr.update_state(labels, predictions)

        precision.update_state(labels, predictions)
        recall.update_state(labels, predictions)

        # region EER
        tp = roc.true_positives.numpy()
        fp = roc.false_positives.numpy()
        tpr = (tp / tp.max()).astype(np.float64)
        fpr = (fp / fp.max()).astype(np.float64)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # endregion

        recall_result = recall.result().numpy()
        precision_result = precision.result().numpy()
        average_precision = -np.sum(np.diff(recall_result) * precision_result[:-1])

        results = {
            "ROC": roc.result(),
            "EER": eer,
            "PR": pr.result(),
            "Precision": average_precision,
        }

        return results

    # endregion

    # region Plotting
    def plot_predictions(self,
                         predictions: List[List[np.ndarray]],
                         labels: np.ndarray,
                         log_dir: str,
                         samples_names: List[str],
                         ):
        metrics_mins = [np.min([np.min(x) for x in metric_pred]) for metric_pred in predictions]
        metrics_maxs = [np.max([np.max(x) for x in metric_pred]) for metric_pred in predictions]

        sample_count = len(labels)
        for i in range(sample_count):
            sample_predictions = [((predictions[j][i] - metrics_mins[j]) / (metrics_maxs[j] - metrics_mins[j]))
                                  for j in range(self.metric_count)]
            self.plot_sample_predictions(predictions=sample_predictions,
                                         labels=labels[i],
                                         log_dir=log_dir,
                                         sample_name=samples_names[i])

    def plot_sample_predictions(self,
                                predictions: List[np.ndarray],
                                labels: np.ndarray,
                                log_dir: str,
                                sample_name: str,
                                linewidth=0.1,
                                include_legend=True,
                                font_size=4,
                                clear_figure=True,
                                ratio=None,
                                ):
        plt.ylim(0.0, 1.0)

        sample_length = len(labels)
        for i in range(self.metric_count):
            metric_predictions = predictions[i]
            if metric_predictions.ndim > 1:
                metric_predictions = np.mean(metric_predictions, axis=tuple(range(1, metric_predictions.ndim)))
            plt.plot(1.0 - metric_predictions, linewidth=linewidth)

        if include_legend:
            plt.legend(self.anomaly_metrics_names,
                       loc="center", bbox_to_anchor=(0.5, -0.4), fontsize=font_size,
                       fancybox=True, shadow=True)

        if ratio is None:

            ratio = np.log(sample_length + 1) * 0.75

        adjust_figure_aspect(plt.gcf(), ratio)

        dpi = (np.sqrt(sample_length) + 100) * 4

        self.plot_labels(labels)

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        labeled_filepath = os.path.join(log_dir, "{}.png".format(sample_name))
        plt.savefig(labeled_filepath, dpi=dpi, bbox_inches='tight')

        if clear_figure:
            plt.clf()

    @staticmethod
    def plot_labels(labels: np.ndarray):
        start = -1

        for i in range(len(labels)):
            if start == -1:
                if labels[i]:
                    start = i
            else:
                if not labels[i]:
                    plt.gca().axvspan(start, i, alpha=0.25, color="red", linewidth=0)
                    start = -1

        if start != -1:
            plt.gca().axvspan(start, len(labels) - 1, alpha=0.25, color="red", linewidth=0)

    # endregion

    def save_evaluation_results(self,
                                log_dir: str,
                                results: Dict[str, List[Union[tf.Tensor, float]]],
                                additional_config: Dict[str, any] = None):
        with open(os.path.join(log_dir, "anomaly_detection_scores.txt"), 'w') as file:
            for i in range(self.metric_count):
                line = "{})".format(self.anomaly_metrics_names[i])
                for result_name, result_values in results.items():
                    value = round(float(result_values[i]), 3)
                    line += " {} = {} |".format(result_name, value)
                line += "\n"
                file.write(line)

            file.write("\n")

            if additional_config is not None:
                for key, value in additional_config.items():
                    file.write("{}: {}\n".format(key, value))

    @property
    def metric_count(self) -> int:
        return len(self.anomaly_metrics_names)


def adjust_figure_aspect(fig, aspect=1.0):
    """
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    """
    x_size, y_size = fig.get_size_inches()
    minsize = min(x_size, y_size)
    x_lim = .4 * minsize / x_size
    y_lim = .4 * minsize / y_size

    if aspect < 1:
        x_lim *= aspect
    else:
        y_lim /= aspect

    fig.subplots_adjust(left=.5 - x_lim,
                        right=.5 + x_lim,
                        bottom=.5 - y_lim,
                        top=.5 + y_lim)


def to_list(x: Union[List, Tuple, Any]) -> Union[List, Tuple]:
    if not (isinstance(x, list) or isinstance(x, tuple)):
        x = [x]
    return x
