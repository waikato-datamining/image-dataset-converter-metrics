import argparse
from typing import Optional, Dict

from wai.logging import LOGGING_WARNING

from idc.metrics.statistic import DatasetStatistic
from ._object_detection_statistic import ObjectDetectionStatisticWithClassMetrics


class BBoxIoU(ObjectDetectionStatisticWithClassMetrics):
    """
    Calculates the accuracy for object detection data.
    """

    def __init__(self, class_metrics: bool = False, iou_threshold: float = None, ignore_labels: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param class_metrics: whether to generate class-level metrics
        :type class_metrics: bool
        :param iou_threshold: the IoU threshold to use
        :type iou_threshold: float
        :param ignore_labels: whether to respect or ignore labels during IoU computation
        :type ignore_labels: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(class_metrics=class_metrics, logger_name=logger_name, logging_level=logging_level)
        self.iou_threshold = iou_threshold
        self.ignore_labels = ignore_labels

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "bbox-iou-od"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the Intersection over Union (= Jaccard Index) for object detection bounding boxes."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-t", "--iou_threshold", type=float, help="The threshold to apply.", default=None, required=False)
        parser.add_argument("-i", "--ignore_labels", action="store_true", help="When set, will ignore labels during IoU computation.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.iou_threshold = ns.iou_threshold
        self.ignore_labels = ns.ignore_labels

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "IoU (BBox)"

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.ignore_labels is None:
            self.ignore_labels = False

    def _initialize_statistic(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        import torchmetrics
        self._statistic = torchmetrics.detection.iou.IntersectionOverUnion(
            box_format="xywh", iou_threshold=self.iou_threshold, class_metrics=self.class_metrics,
            respect_labels=not self.ignore_labels)

    def _post_process(self, stat: DatasetStatistic, meta: Optional[Dict[str, int]] = None):
        """
        For post-processing the statistic.

        :param stat: the statistic to post-process
        :type stat: DatasetStatistic
        :param meta: the class lookup
        :type meta: dict
        """
        if "/cl_" in stat.statistic:
            try:
                idx = int(stat.statistic[stat.statistic.rindex("_")+1:])
                for k, v in meta.items():
                    if v == idx:
                        stat.statistic = stat.statistic[0:stat.statistic.index("/")] + "/" + k
                        break
            except:
                self.logger().warning("Failed to extract class index from: %s" % stat.statistic)
