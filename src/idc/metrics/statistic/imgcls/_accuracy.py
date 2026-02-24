import argparse
from typing import List

from wai.logging import LOGGING_WARNING

from ._classification_statistic import ClassificationStatisticWithAverage


class Accuracy(ClassificationStatisticWithAverage):
    """
    Calculates the accuracy for image classification data.
    """

    def __init__(self, num_classes: int = None, average: str = None, top_k: int = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param num_classes: the number of classes
        :type num_classes: int
        :param average: the average to use
        :type average: str
        :param top_k: only compute the accuracy for the top K classes
        :type top_k: int
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(num_classes=num_classes, average=average, logger_name=logger_name, logging_level=logging_level)
        self.top_k = top_k

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "accuracy-ic"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the accuracy for image classification data."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-k", "--top_k", type=int, help="Only take the top K classes into account.", default=1, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.top_k = ns.top_k

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "Accuracy"

    def _initialize_statistic(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        import torchmetrics
        if self.top_k is not None:
            self._statistic = torchmetrics.Accuracy(task="multiclass", average=self.average, num_classes=self.num_classes, top_k=self.top_k)
        else:
            self._statistic = torchmetrics.Accuracy(task="multiclass", average=self.average, num_classes=self.num_classes)
