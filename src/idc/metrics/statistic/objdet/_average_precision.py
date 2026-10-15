import argparse
import contextlib
import io
from typing import List, Union

from wai.logging import LOGGING_WARNING
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ._object_detection_statistic import MSCOCOObjectDetectionStatistic
from idc.metrics.statistic import DatasetStatistic, DatasetStatisticList

STATISTIC_AP = "AP"
STATISTIC_AP50 = "AP50"
STATISTIC_AP75 = "AP75"
STATISTIC_APs = "APs"
STATISTIC_APm = "APm"
STATISTIC_APl = "APl"
STATISTIC_AR1 = "AR1"
STATISTIC_AR10 = "AR10"
STATISTIC_AR100 = "AR100"
STATISTIC_ARs = "ARs"
STATISTIC_ARm = "ARm"
STATISTIC_ARl = "ARl"
STATISTICS = [
    STATISTIC_AP,
    STATISTIC_AP50,
    STATISTIC_AP75,
    STATISTIC_APs,
    STATISTIC_APm,
    STATISTIC_APl,
    STATISTIC_AR1,
    STATISTIC_AR10,
    STATISTIC_AR100,
    STATISTIC_ARs,
    STATISTIC_ARm,
    STATISTIC_ARl,
]

IOU_TYPE_BBOX = "bbox"
IOU_TYPE_SEGMENTATION = "segm"
IOU_TYPES = [
    IOU_TYPE_BBOX,
    IOU_TYPE_SEGMENTATION,
]


class AveragePrecision(MSCOCOObjectDetectionStatistic):
    """
    Calculates the average precision (AP) for object detection data.
    """

    def __init__(self, statistics: List[str] = None, iou_type: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param statistics: the statistics to generate
        :type statistics: list
        :param iou_type: how to evaluate
        :type iou_type: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.statistics = statistics
        self.iou_type = iou_type

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "average-precision-od"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the selected average precision (AP) statistics for object detection data. Retrieves the 'score' meta-data value from the object detection predictions, uses 1.0 if not present."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-s", "--statistic", choices=STATISTICS, help="The statistic(s) to compute.", default=None, required=True, nargs="+")
        parser.add_argument("-t", "--iou_type", choices=IOU_TYPES, help="How to evaluate the intersect-over-union (IoU).", default=IOU_TYPE_SEGMENTATION, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.statistics = ns.statistic
        self.iou_type = ns.iou_type

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "AP"

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if (self.statistics is None) or (len(self.statistics) == 0):
            self.statistics = [STATISTIC_AP]
        if self.iou_type is None:
            self.iou_type = IOU_TYPE_SEGMENTATION

    def calculate(self, anns, preds, meta=None) -> Union[DatasetStatistic, DatasetStatisticList]:
        """
        Calculates the statistic from the tensors with annotations and predictions.

        :param anns: the list of annotations (dict of tensors/labels)
        :param preds: the list of predictions (dict of tensors/labels)
        :param meta: optional meta-data that is required for the calculation (domain-specific)
        :return: the generated statistic(s)
        :rtype: DatasetStatistic or DatasetStatisticList
        """
        # redirect stdout into unused buffer, to suppress coco output
        with contextlib.redirect_stdout(io.StringIO()):
            coco_anns = COCO()
            coco_anns.dataset = anns.data
            coco_anns.createIndex()

            coco_preds = COCO()
            coco_preds.dataset = preds.data
            coco_preds.createIndex()

            coco_eval = COCOeval(coco_anns, coco_preds, iouType=self.iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = []
        for stat in self.statistics:
            if stat == STATISTIC_AP:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[0]))
            elif stat == STATISTIC_AP50:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[1]))
            elif stat == STATISTIC_AP75:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[2]))
            elif stat == STATISTIC_APs:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[3]))
            elif stat == STATISTIC_APm:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[4]))
            elif stat == STATISTIC_APl:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[5]))
            elif stat == STATISTIC_AR1:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[6]))
            elif stat == STATISTIC_AR10:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[7]))
            elif stat == STATISTIC_AR100:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[8]))
            elif stat == STATISTIC_ARs:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[9]))
            elif stat == STATISTIC_ARm:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[10]))
            elif stat == STATISTIC_ARl:
                stats.append(DatasetStatistic(statistic=stat, value=coco_eval.stats[11]))
            else:
                raise Exception("Unhandled statistic: %s" % stat)
        if len(stats) == 0:
            raise Exception("No statistics calculated?")
        if len(stats) == 1:
            return stats[0]
        return DatasetStatisticList(stats)
