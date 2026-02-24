import abc
import argparse
import logging
from typing import List

from wai.logging import LOGGING_WARNING

from idc.metrics.api import ImagePairList
from kasperl.api import make_list
from idc.metrics.statistic import DatasetStatisticFilter, DatasetStatistic


def determine_classes(data: ImagePairList, logger: logging.Logger = None):
    """
    Processes the image pairs and returns the classes and the lookup.

    :param data: the image pairs to use
    :type data: ImagePairList
    :param logger: optional logger instance to use
    :type logger: logging.Logger
    :return: the tuple of: annotation classes, predictions classes and the class lookup
    :rtype: tuple
    """
    from torch import tensor

    classes = set()
    for pair in make_list(data):
        if pair.annotation.has_annotation() and pair.prediction.has_annotation():
            classes.add(pair.annotation.annotation)
            classes.add(pair.prediction.annotation)
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
            if pair.annotation.has_annotation() and pair.prediction.has_annotation():
                anns.append(lookup[pair.annotation.annotation])
                preds.append(lookup[pair.prediction.annotation])
        anns = tensor(anns)
        preds = tensor(preds)
        return anns, preds, lookup

    return None, None, None


class NumClassesHandler:
    """
    Mixin for classes that require to know the number of classes.
    """

    def set_num_classes(self, num_classes: int):
        """
        Sets the number of classes to use.

        :param num_classes: the number of classes
        :type num_classes: int
        """
        raise NotImplementedError()


class ClassificationStatistic(DatasetStatisticFilter, NumClassesHandler, abc.ABC):
    """
    Ancestor for classification statistics.
    """

    def __init__(self, num_classes: int = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param num_classes: the number of classes
        :type num_classes: int
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.num_classes = num_classes
        self._statistic = None

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-n", "--num_classes", type=int, help="The number of classes in the dataset.", default=None, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.num_classes = ns.num_classes

    def set_num_classes(self, num_classes: int):
        """
        Sets the number of classes to use.

        :param num_classes: the number of classes
        :type num_classes: int
        """
        self.num_classes = num_classes

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

    def calculate(self, anns, preds) -> DatasetStatistic:
        """
        Calculates the statistic from the tensors with annotations and predictions.

        :param anns: the tensor with the class label indices of the annotations
        :param preds: the tensor with the class label indices of the predictions
        :return: the generated statistic
        :rtype: DatasetStatistic
        """
        if self._statistic is None:
            self._initialize_statistic()
        result = DatasetStatistic(statistic=self._statistic_name())
        result.value = float(self._statistic(anns, preds))
        return result

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the statistic
        """
        result = None
        anns, preds, lookup = determine_classes(data)
        if (anns is not None) and (preds is not None):
            result = self.calculate(preds, anns)

        return result


class ClassificationStatisticWithAverage(ClassificationStatistic, abc.ABC):
    """
    Ancestor for classification statistics that support averages.
    """

    def __init__(self, num_classes: int = None, average: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param num_classes: the number of classes
        :type num_classes: int
        :param average: the average to use
        :type average: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(num_classes=num_classes, logger_name=logger_name, logging_level=logging_level)
        self.average = average

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-a", "--average", choices=self._averages(), help="The average to use.", default=self._default_average(), required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.average = ns.average

    def _averages(self) -> List[str]:
        """
        Returns the possible averages.

        :return: the averages
        :rtype: list
        """
        return ["micro", "macro", "weighted", "none"]

    def _default_average(self) -> str:
        """
        Returns the default average.

        :return: the default
        :rtype: str
        """
        return "micro"
