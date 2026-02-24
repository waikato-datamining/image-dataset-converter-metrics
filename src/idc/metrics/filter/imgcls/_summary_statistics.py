import argparse
from typing import List

from wai.logging import LOGGING_WARNING

from build.lib.seppl import SessionHandler
from idc.metrics.api import ImagePairList
from idc.metrics.registry import available_imgcls_statistics
from idc.metrics.statistic import DatasetStatisticList
from idc.metrics.statistic.imgcls import ClassificationStatistic
from idc.metrics.statistic.imgcls import NumClassesHandler, determine_classes
from seppl import split_args, Plugin, Initializable, init_initializable, split_cmdline
from seppl.io import BatchFilter


class SummaryStatistics(BatchFilter):
    """
    Calculates summary statistics for the incoming data pairs.
    """

    def __init__(self, statistics: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param statistics: the statistics (and their options) to generate
        :type statistics: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.statistics = statistics
        self._statistics = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "summary-statistics-ic"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates summary statistics for the incoming data pairs."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImagePairList]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [DatasetStatisticList]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-s", "--statistics", type=str, default=None, help="The summary statistics to calculate.", required=True)
        return parser

    def _parse_statistics(self) -> List[Plugin]:
        """
        Parses the statistics command-line and returns the list of plugins it represents.
        Raises an exception in case of an invalid statistic.
        
        :return: the list of plugins
        :rtype: list
        """
        from seppl import args_to_objects

        # split command-line into valid plugin subsets
        valid = dict()
        valid.update(available_imgcls_statistics())
        stats = split_cmdline(self.statistics)
        args = split_args(stats, list(valid.keys()))
        return args_to_objects(args, valid, allow_global_options=False)

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.statistics = ns.statistics

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        if self.statistics is None:
            raise Exception("No statistics defined!")

        self._statistics = self._parse_statistics()

        for statistic in self._statistics:
            if not isinstance(statistic, ClassificationStatistic):
                raise Exception("Not a classification statistic: %s" % str(type(statistic)))
            if isinstance(statistic, SessionHandler):
                statistic.session = self.session
            if isinstance(statistic, Initializable):
                init_initializable(statistic, "statistic")

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        anns, preds, lookup = determine_classes(data, logger=self.logger())
        result = DatasetStatisticList()
        for statistic in self._statistics:
            if isinstance(statistic, NumClassesHandler):
                statistic.set_num_classes(len(lookup))
            if isinstance(statistic, ClassificationStatistic):
                try:
                    stat = statistic.calculate(anns, preds)
                    result.append(stat)
                except:
                    self.logger().exception("Failed to obtain statistic: %s" % str(type(statistic)))
            else:
                raise Exception("Unhandled type of statistic: %s" % str(type(statistic)))

        return result
