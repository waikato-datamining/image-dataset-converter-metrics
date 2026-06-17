import abc
import argparse
import logging
from typing import Dict, Optional, Union, Tuple, List

from wai.logging import LOGGING_WARNING

from idc.api import get_object_label
from idc.metrics.api import ImagePairList
from idc.metrics.statistic import DatasetStatisticFilter, DatasetStatistic, DatasetStatisticList
from kasperl.api import make_list


def prepare_data(data: ImagePairList, logger: logging.Logger = None) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[Dict[str, int]]]:
    """
    Processes the image pairs and returns the tuple of annotation bbox tensors, prediction bbox tensors and class lookup.
    The bbox tensors are in xywh format.

    list of dicts, one per image

    :param data: the image pairs to use
    :type data: ImagePairList
    :param logger: optional logger instance to use
    :type logger: logging.Logger
    :return: the tuple of annotation dict, prediction dict, class lookup; the ann/pred dicts are of format {"boxes": tensor, "labels": tensor}
    :rtype: tuple
    """
    from torch import tensor

    classes = set()
    for pair in make_list(data):
        if pair.annotation.has_annotation() and pair.prediction.has_annotation():
            for lobj in pair.annotation.annotation:
                classes.add(get_object_label(lobj))
            for lobj in pair.prediction.annotation:
                classes.add(get_object_label(lobj))
    classes = sorted(list(classes))
    if logger is not None:
        logger.info("%d classes: %s" % (len(classes), ", ".join(classes)))

    if len(classes) > 0:
        lookup = dict()
        for i, cls in enumerate(classes):
            lookup[cls] = i
        anns = []
        preds = []
        for pair in make_list(data):
            anns_boxes = []
            anns_labels = []
            preds_boxes = []
            preds_labels = []
            preds_scores = []
            if pair.annotation.has_annotation() and pair.prediction.has_annotation():
                for lobj in pair.annotation.annotation:
                    anns_boxes.append([lobj.x, lobj.y, lobj.width, lobj.height])
                    anns_labels.append(lookup[get_object_label(lobj)])  # use index
                for lobj in pair.prediction.annotation:
                    preds_boxes.append([lobj.x, lobj.y, lobj.width, lobj.height])
                    preds_labels.append(lookup[get_object_label(lobj)])  # use index
                    preds_scores.append(float(lobj.metadata["score"]) if ("score" in lobj.metadata) else 1.0)
                anns.append({"boxes": tensor(anns_boxes), "labels": tensor(anns_labels)})
                preds.append({"boxes": tensor(preds_boxes), "labels": tensor(preds_labels), "scores": tensor(preds_scores)})

        return anns, preds, lookup

    return None, None, None


class ObjectDetectionStatistic(DatasetStatisticFilter, abc.ABC):
    """
    Ancestor for object detection statistics.
    """

    def __init__(self, logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self._statistic = None
        self._class_lookup = None

    def _initialize_statistic(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        raise NotImplementedError()

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        raise NotImplementedError()

    def _post_process(self, stat: DatasetStatistic, meta: Optional[Dict[str, int]] = None):
        """
        For post-processing the statistic.

        :param stat: the statistic to post-process
        :type stat: DatasetStatistic
        :param meta: the class lookup
        :type meta: dict
        """
        pass

    def calculate(self, anns, preds, meta: Optional[Dict[str, int]] = None) -> Union[DatasetStatistic, DatasetStatisticList]:
        """
        Calculates the statistic from the tensors with annotations and predictions.

        :param anns: the list of annotations (dict of tensors/labels)
        :param preds: the list of predictions (dict of tensors/labels)
        :param meta: optional meta-data that is required for the calculation (domain-specific)
        :return: the generated statistic(s)
        :rtype: DatasetStatistic or DatasetStatisticList
        """
        if self._statistic is None:
            self._initialize_statistic()
        stat = self._statistic(preds, anns)
        if isinstance(stat, dict):
            result = DatasetStatisticList()
            for k in stat:
                s = DatasetStatistic(statistic=k)
                t = stat[k]
                if len(t.shape) == 0:
                    s.value = float(t)
                elif len(t.shape) == 1:
                    s.value = float(t[0])
                else:
                    s.value = t
                self._post_process(s, meta=meta)
                result.append(s)
        else:
            result = DatasetStatistic(statistic=self._statistic_name(), value=stat)
            self._post_process(result, meta=meta)
        return result

    def _prepare_data(self, data) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Dict[str, int]]:
        """
        Processes the image pairs and returns the tuple of annotation bbox tensors, prediction bbox tensors and class lookup.
        The bbox tensors are in xywh format.

        list of dicts, one per image

        :param data: the image pairs to use
        :type data: ImagePairList
        :return: the tuple of annotation dict, prediction dict, class lookup; the ann/pred dicts are of format {"boxes": tensor, "labels": tensor}
        :rtype: tuple
        """
        return prepare_data(data, logger=self.logger())

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the statistic
        """
        result = None
        anns, preds, lookup = self._prepare_data(data)
        if (anns is not None) and (preds is not None):
            result = self.calculate(preds, anns, meta=lookup)

        return result


class ObjectDetectionStatisticWithClassMetrics(ObjectDetectionStatistic, abc.ABC):
    """
    Ancestor for classification statistics that offer class metrics.
    """

    def __init__(self, class_metrics: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param class_metrics: whether to generate class-level metrics
        :type class_metrics: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.class_metrics = class_metrics
        self._statistic = None

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-c", "--class_metrics", action="store_true", help="Whether to generate class-level metrics.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.class_metrics = ns.class_metrics

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.class_metrics is None:
            self.class_metrics = False
