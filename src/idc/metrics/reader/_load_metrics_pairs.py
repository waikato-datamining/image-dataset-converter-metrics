import argparse
from typing import List, Iterable, Tuple, Optional, Dict

from wai.logging import LOGGING_WARNING

from idc.api import ImageData
from idc.metrics.api import ImagePair, ImagePairList
from idc.registry import available_readers, available_filters
from kasperl.api import PIPELINE_FORMATS, PIPELINE_FORMAT_CMDLINE, load_pipeline
from kasperl.api import Reader
from seppl import Plugin, split_args, Initializable, init_initializable
from seppl.io import BatchFilter, MultiFilter, Filter


class LoadMetricsPairsReader(Reader):

    def __init__(self, annotations_subflow: str = None, annotations_flow_format: str = None,
                 predictions_subflow: str = None, predictions_flow_format: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the reader.

        :param annotations_subflow: the sub-flow for reading the annotations
        :type annotations_subflow: str
        :param predictions_subflow: the sub-flow for reading the predictions
        :type predictions_subflow: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.annotations_subflow = annotations_subflow
        self.annotations_flow_format = annotations_flow_format
        self.predictions_subflow = predictions_subflow
        self.predictions_flow_format = predictions_flow_format
        self._annotations_subflow = None
        self._annotations_reader = None
        self._annotations_filter = None
        self._predictions_subflow = None
        self._predictions_reader = None
        self._predictions_filter = None
        self._common_names = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "load-metrics-pairs"

    def description(self) -> str:
        """
        Returns a description of the reader.

        :return: the description
        :rtype: str
        """
        return "Loads the annotation/prediction pairs using the respective sub-flows and forwards matching pairs for calculating metrics."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-a", "--annotations_flow", type=str, default=None, help="The subflow to loading the annotations (reader and optional filter(s)).")
        parser.add_argument("-A", "--annotations_flow_format", choices=PIPELINE_FORMATS, default=PIPELINE_FORMAT_CMDLINE, help="The format of the annotations pipeline.")
        parser.add_argument("-p", "--predictions_flow", type=str, default=None, help="The subflow to loading the predictions (reader and optional filter(s)).")
        parser.add_argument("-P", "--predictions_flow_format", choices=PIPELINE_FORMATS, default=PIPELINE_FORMAT_CMDLINE, help="The format of the predictions pipeline.")
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.annotations_flow = ns.annotations_flow
        self.annotations_flow_format = ns.annotations_flow_format
        self.predictions_flow = ns.predictions_flow
        self.predictions_flow_format = ns.predictions_flow_format

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImagePair]

    def _parse_sub_flow(self, flow: str, flow_format: str) -> List[Plugin]:
        """
        Parses the command-line and returns the list of plugins it represents.
        Raises an exception in case of an invalid sub-flow.

        :param flow: the flow to parse
        :type flow: str
        :param flow_format: the format of the flow
        :type flow_format: str
        :return: the list of plugins
        :rtype: list
        """
        from seppl import args_to_objects

        # split command-line into valid plugin subsets
        valid = dict()
        valid.update(available_readers())
        valid.update(available_filters())
        pipeline = load_pipeline(flow, flow_format, logger=self.logger())
        args = split_args(pipeline, list(valid.keys()))
        return args_to_objects(args, valid, allow_global_options=False)

    def _initialize_sub_flow(self, sub_flow: List[Plugin]) -> Tuple[Optional[Reader], Optional[Filter]]:
        """
        Initializes the sub-flow.
        
        :param sub_flow: the sub-flow plugins to initialize
        :type sub_flow: list 
        :return: the tuple of reader/filter
        :rtype: tuple
        """
        _reader = None
        _filter = None
        if len(sub_flow) > 0:
            filters = []
            for plugin in sub_flow:
                if isinstance(plugin, Reader):
                    if len(filters) > 0:
                        raise Exception("Reader must be first plugin in sub-flow!")
                    _reader = plugin
                if isinstance(plugin, BatchFilter):
                    filters.append(plugin)
            if len(filters) == 1:
                _filter = filters[0]
            elif len(filters) > 1:
                _filter = MultiFilter(filters=filters)

        if _reader is not None:
            _reader.session = self.session
            if isinstance(_reader, Initializable):
                init_initializable(_reader, "writer")
        if _filter is not None:
            _filter.session = self.session
            if isinstance(_filter, Initializable):
                init_initializable(_filter, "filter")
                
        return _reader, _filter

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        self._common_names = set()

        if self.annotations_flow is None:
            raise Exception("No annotations sub-flow specified!")
        if self.annotations_flow_format is None:
            self.annotations_flow_format = PIPELINE_FORMAT_CMDLINE
        self._annotations_subflow = self._parse_sub_flow(self.annotations_flow, self.annotations_flow_format)
        self._annotations_reader, self._annotations_filter = self._initialize_sub_flow(self._annotations_subflow)
        if self._annotations_reader is None:
            raise Exception("No annotations reader specified!")

        if self.predictions_flow is None:
            raise Exception("No predictions sub-flow specified!")
        if self.predictions_flow_format is None:
            self.predictions_flow_format = PIPELINE_FORMAT_CMDLINE
        self._predictions_subflow = self._parse_sub_flow(self.predictions_flow, self.predictions_flow_format)
        self._predictions_reader, self._predictions_filter = self._initialize_sub_flow(self._predictions_subflow)
        if self._predictions_reader is None:
            raise Exception("No predictions reader specified!")

    def _create_lookup(self, items: List[ImageData]) -> Dict[str, ImageData]:
        """
        Generates a lookup from the list of items.

        :param items: the items to turn into a dict
        :type items: list
        :return: the generated lookup, using image_name as key
        :rtype: dict
        """
        result = dict()
        for item in items:
            result[item.image_name] = item
        return result

    def read(self) -> Iterable:
        """
        Loads the data and returns the items one by one.

        :return: the data
        :rtype: Iterable
        """
        self.logger().info("Reading annotations...")
        annotations = []
        while not self._annotations_reader.has_finished():
            for item in self._annotations_reader.read():
                if item is not None:
                    annotations.append(item)
        if self._annotations_filter is not None:
            annotations = self._annotations_filter.process(annotations)
        self.logger().info("# annotations: %d" % len(annotations))
        annotations_lookup = self._create_lookup(annotations)

        self.logger().info("Reading predictions...")
        predictions = []
        while not self._predictions_reader.has_finished():
            for item in self._predictions_reader.read():
                if item is not None:
                    predictions.append(item)
        if self._predictions_filter is not None:
            predictions = self._predictions_filter.process(predictions)
        self.logger().info("# predictions: %d" % len(predictions))
        predictions_lookup = self._create_lookup(predictions)

        self._common_names = list(set(annotations_lookup.keys()) & set(predictions_lookup.keys()))
        self.logger().info("# pairs: %d" % len(self._common_names))
        if len(self._common_names) == 0:
            return None

        result = ImagePairList()
        for image_name in self._common_names:
            if type(annotations_lookup[image_name]) is not type(predictions_lookup[image_name]):
                raise Exception("Annotation and prediction differ in type: %s != %s"
                                % (str(type(annotations_lookup[image_name])), str(type(predictions_lookup[image_name]))))
            result.append(ImagePair(
                image_name=image_name,
                annotation=annotations_lookup[image_name],
                prediction=predictions_lookup[image_name]))
        yield result

        self._common_names = None
        return None

    def has_finished(self) -> bool:
        """
        Returns whether reading has finished.

        :return: True if finished
        :rtype: bool
        """
        return True
