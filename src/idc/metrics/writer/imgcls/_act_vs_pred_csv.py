import argparse
import csv
from typing import List, Iterable

from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData
from idc.metrics.api import ImagePairList, ImagePair
from kasperl.api import BatchWriter
from seppl.placeholders import placeholder_list, PlaceholderSupporter


class ActualVsPredictedCSVWriter(BatchWriter, PlaceholderSupporter):

    def __init__(self, output_file: str = None, column_image: str = None, column_actual: str = None, column_predicted: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the writer.

        :param output_file: the file to write the actual vs predicted data to
        :type output_file: str
        :param column_image: the column name for the image name
        :type column_image: str
        :param column_actual: the column name for the actual values
        :type column_actual: str
        :param column_predicted: the column name for the predicted values
        :type column_predicted: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.output_file = output_file
        self.column_image = column_image
        self.column_actual = column_actual
        self.column_predicted = column_predicted

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "to-act-vs-pred-ic"

    def description(self) -> str:
        """
        Returns a description of the writer.

        :return: the description
        :rtype: str
        """
        return "Outputs a CSV file with actual vs predicted columns."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-o", "--output", type=str, help="The CSV file to store the actual vs predicted data in. " + placeholder_list(obj=self), required=True)
        parser.add_argument("-i", "--column_image", type=str, help="The column name for the image name.", required=False, default="Image")
        parser.add_argument("-a", "--column_actual", type=str, help="The column name for the actual values.", required=False, default="Actual")
        parser.add_argument("-p", "--column_predicted", type=str, help="The column name for the predicted values.", required=False, default="Predicted")
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.output_file = ns.output
        self.column_image = ns.column_image
        self.column_actual = ns.column_actual
        self.column_predicted = ns.column_predicted

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImagePair]

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        if self.output_file is None:
            raise Exception("No output file specified!")

        if self.column_image is None:
            self.column_image = "Image"
        if self.column_actual is None:
            self.column_actual = "Actual"
        if self.column_predicted is None:
            self.column_predicted = "Predicted"

    def write_batch(self, data: Iterable):
        """
        Saves the data in one go.

        :param data: the data to write
        :type data: Iterable
        """
        for item in data:
            if not isinstance(item, ImagePairList):
                self.logger().warning("Unhandled data type: %s" % str(type(item)))
                continue

            rows = [[self.column_image, self.column_actual, self.column_predicted]]
            for pair in item:
                if isinstance(pair.annotation, ImageClassificationData):
                    rows.append([pair.image_name, pair.annotation.annotation, pair.prediction.annotation])

            path = self.session.expand_placeholders(self.output_file)
            self.logger().info("Writing data to: %s" % path)
            with open(path, "w") as fp:
                writer = csv.writer(fp, quoting=csv.QUOTE_MINIMAL)
                writer.writerows(rows)
