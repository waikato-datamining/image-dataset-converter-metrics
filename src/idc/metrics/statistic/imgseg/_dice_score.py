import argparse

from wai.logging import LOGGING_WARNING

from idc.metrics.statistic import NumClassesHandler
from ._image_segmentation_statistic import ImageSegmentationStatistic

AVERAGE_MICRO = "micro"
AVERAGE_MACRO = "macro"
AVERAGE_WEIGHTED = "weighted"
AVERAGE_NONE = "none"
AVERAGES = [
    AVERAGE_MICRO,
    AVERAGE_MACRO,
    AVERAGE_WEIGHTED,
    AVERAGE_NONE,
]

AGGREGATION_SAMPLEWISE = "samplewise"
AGGREGATION_GLOBAL = "global"
AGGREGATION_TYPES = [
    AGGREGATION_SAMPLEWISE,
    AGGREGATION_GLOBAL,
]


class DiceScore(ImageSegmentationStatistic, NumClassesHandler):
    """
    Calculates the mean intersection over union (mIoU) for image segmentation data.
    """

    def __init__(self, num_classes: int = None, average: str = None, aggregation_level: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param num_classes: the number of classes
        :type num_classes: int
        :param average: the type of average to compute
        :type average: str
        :param aggregation_level: the aggregation level to use
        :type aggregation_level: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.num_classes = num_classes
        self.average = average
        self.aggregation_level = aggregation_level

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "dice-score-is"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the dice score for image segmentation data."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-n", "--num_classes", type=int, help="The number of classes in the dataset.", default=None, required=False)
        parser.add_argument("-a", "--average", choices=AVERAGES, help="The type of average to compute.", default=AVERAGE_MACRO, required=False)
        parser.add_argument("-g", "--aggregation_level", choices=AGGREGATION_TYPES, help="The aggregation level to use.", default=AGGREGATION_SAMPLEWISE, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.num_classes = ns.num_classes
        self.average = ns.average
        self.aggregation_level = ns.aggregation_level

    def set_num_classes(self, num_classes: int):
        """
        Sets the number of classes to use.

        :param num_classes: the number of classes
        :type num_classes: int
        """
        self.num_classes = num_classes

    def initialize(self):
        super().initialize()
        if self.average is None:
            self.average = AVERAGE_MACRO
        if self.aggregation_level is None:
            self.aggregation_level = AGGREGATION_SAMPLEWISE

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "dice-score"

    def _initialize_statistic(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        import torchmetrics.segmentation
        self._statistic = torchmetrics.segmentation.DiceScore(
            self.num_classes, include_background=not self.exclude_background, average=self.average,
            aggregation_level=self.aggregation_level, input_format="one-hot")
