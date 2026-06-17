import argparse

from wai.logging import LOGGING_WARNING

from ._object_detection_statistic import ObjectDetectionStatisticWithClassMetrics

AVERAGE_MICRO = "micro"
AVERAGE_MACRO = "macro"
AVERAGES = [
    AVERAGE_MICRO,
    AVERAGE_MACRO
]

BACKEND_PYCOCOTOOLS = "pycocotools"
BACKEND_FASTER_COCO_EVAL = "faster-coco-eval"
BACKENDS = [
    BACKEND_PYCOCOTOOLS,
    BACKEND_FASTER_COCO_EVAL,
]


class MeanAveragePrecision(ObjectDetectionStatisticWithClassMetrics):
    """
    Calculates the mean average precision (mAP) for object detection data.
    """

    def __init__(self, class_metrics: bool = False, average: str = None, backend: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param class_metrics: whether to generate class-level metrics
        :type class_metrics: bool
        :param average: the type of average to compute
        :type average: str
        :param backend: the backend to use for calculation
        :type backend: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(class_metrics=class_metrics, logger_name=logger_name, logging_level=logging_level)
        self.average = average
        self.backend = backend

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "mean-average-precision-od"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the mean average precision (mAP) for object detection data. Retrieves the 'score' meta-data value from the object detection predictions, uses 1.0 if not present."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-a", "--average", choices=AVERAGES, help="The type of average to compute.", default=AVERAGE_MICRO, required=False)
        parser.add_argument("-b", "--backend", choices=BACKENDS, help="The type of backend to use for computation.", default=BACKEND_PYCOCOTOOLS, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.average = ns.average
        self.backend = ns.backend

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "mAP"

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.average is None:
            self.average = AVERAGE_MICRO
        if self.backend is None:
            self.backend = BACKEND_PYCOCOTOOLS

    def _initialize_statistic(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        import torchmetrics
        self._statistic = torchmetrics.detection.mean_ap.MeanAveragePrecision(
            box_format="xywh", iou_type="bbox", class_metrics=self.class_metrics,
            average=self.average, backend=self.backend)
