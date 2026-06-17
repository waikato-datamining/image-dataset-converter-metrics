import abc
import argparse
import logging
import traceback
from typing import Dict, Optional, Union, Tuple, List, Any

import numpy as np
from wai.logging import LOGGING_WARNING

from idc.metrics.api import ImagePairList
from idc.metrics.statistic import DatasetStatisticFilter, DatasetStatistic, DatasetStatisticList
from kasperl.api import make_list


def prepare_data(data: ImagePairList, logger: logging.Logger = None) -> Tuple[Any, Any, Optional[Dict[str, int]]]:
    """
    Processes the image pairs and returns the tuple of annotation bbox tensors, prediction bbox tensors and class lookup.

    :param data: the image pairs to use
    :type data: ImagePairList
    :param logger: optional logger instance to use
    :type logger: logging.Logger
    :return: the tuple of annotation tensor, prediction tensor, class lookup; the ann/pred tensors are of format N,C,W,H (N=#images, C=#classes, W=width, H=height)
    :rtype: tuple
    """
    import torch

    classes = set()
    for pair in make_list(data):
        if pair.annotation.has_annotation() and pair.prediction.has_annotation():
            classes.update(pair.annotation.annotation.labels)
            classes.update(pair.prediction.annotation.labels)
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
            anns_layers = []
            preds_layers = []
            if pair.annotation.has_annotation() and pair.prediction.has_annotation():
                for cls in classes:
                    if cls in pair.annotation.annotation.layers:
                        anns_layers.append(pair.annotation.annotation.layers[cls])
                    else:
                        anns_layers.append(np.zeros(pair.annotation.image_size, dtype=np.uint8))
                    if cls in pair.prediction.annotation.layers:
                        preds_layers.append(pair.prediction.annotation.layers[cls])
                    else:
                        preds_layers.append(np.zeros(pair.annotation.image_size, dtype=np.uint8))
            anns.append(anns_layers)
            preds.append(preds_layers)

        try:
            anns_tensor = torch.as_tensor(np.array(anns, dtype=np.uint8))
            preds_tensor = torch.as_tensor(np.array(preds, dtype=np.uint8))
            return anns_tensor, preds_tensor, lookup
        except:
            if logger is not None:
                logger.exception("Failed to create tensors!")
            else:
                print("Failed to create tensors!")
                traceback.print_exc()

    return None, None, None


class ImageSegmentationStatistic(DatasetStatisticFilter, abc.ABC):
    """
    Ancestor for image segmentation statistics.
    """

    def __init__(self, exclude_background: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param exclude_background: whether to exclude the background (= 1st layer) from the calculation
        :type exclude_background: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.exclude_background = exclude_background
        self._statistic = None

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-b", "--exclude_background", action="store_true", help="Whether to exclude the background (= 1st layer) from the calculation.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.exclude_background = ns.exclude_background

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.exclude_background is None:
            self.exclude_background = False

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
        if len(stat.shape) == 0:
            stat = float(stat)
        elif len(stat.shape) == 1:
            stat = float(stat[0])
        else:
            self.logger().warning("Unhandled statistic type: %s" % str(type(stat)))
        result = DatasetStatistic(statistic=self._statistic_name(), value=stat)
        self._post_process(result, meta=meta)
        return result

    def _prepare_data(self, data) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Dict[str, int]]:
        """
        Processes the image pairs and returns the tuple of annotation tensor, prediction tensor and class lookup.

        list of dicts, one per image

        :param data: the image pairs to use
        :type data: ImagePairList
        :return: the tuple of annotation tensor, prediction tensor, class lookup; the ann/pred tensors are of format N,C,W,H (N=#images, C=#classes, W=width, H=height)
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


class ImageSegmentationStatisticWithPerClass(ImageSegmentationStatistic, abc.ABC):
    """
    Ancestor for classification statistics that offer class metrics.
    """

    def __init__(self, exclude_background: bool = False, per_class: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param exclude_background: whether to exclude the background (= 1st layer) from the calculation
        :type exclude_background: bool
        :param per_class: whether to generate per class metrics
        :type per_class: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(exclude_background=exclude_background, logger_name=logger_name, logging_level=logging_level)
        self.per_class = per_class
        self._statistic = None

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-c", "--per_class", action="store_true", help="Whether to generate per class metrics.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.per_class = ns.per_class

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.per_class is None:
            self.per_class = False
