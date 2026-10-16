import abc
import logging
from typing import Dict, Optional, Union, Tuple, Any

from idc.api import COCOAnnotations
from idc.metrics.api import ImagePairList
from idc.metrics.statistic import DatasetStatisticFilter, DatasetStatistic, DatasetStatisticList
from kasperl.api import make_list


def prepare_mscoco_data(data: ImagePairList, logger: logging.Logger = None) -> Tuple[Optional[COCOAnnotations], Optional[COCOAnnotations], Optional[Dict[str, Any]]]:
    """
    Processes the image pairs and returns the tuple of MSCOCO annotations, MSCOCO predictions, class lookup and image lookup.

    list of dicts, one per image

    :param data: the image pairs to use
    :type data: ImagePairList
    :param logger: optional logger instance to use
    :type logger: logging.Logger
    :return: the tuple of annotations, predictions, class lookup, image lookup
    :rtype: tuple
    """
    classes = set()
    for pair in make_list(data):
        if pair.annotation.has_annotation() and pair.prediction.has_annotation():
            classes.update(pair.annotation.annotation.layers.keys())
            classes.update(pair.prediction.annotation.layers.keys())
    classes = sorted(list(classes))
    if logger is not None:
        logger.info("%d classes: %s" % (len(classes), ", ".join(classes)))

    if len(classes) > 0:
        image_lookup = dict()
        category_lookup = dict()
        for i, cls in enumerate(classes):
            category_lookup[cls] = i
        anns = COCOAnnotations(image_lookup, category_lookup)
        preds = COCOAnnotations(image_lookup, category_lookup)
        for pair in make_list(data):
            if pair.annotation.has_annotation() and pair.prediction.has_annotation():
                anns.append_image(pair.annotation)
                anns.append_annotations(pair.annotation)
                preds.append_image(pair.prediction)
                preds.append_annotations(pair.prediction)

        anns.finalize()
        preds.finalize()
        return anns, preds, {"categories": category_lookup, "images": image_lookup}

    return None, None, None


class ImageSegmentationStatistic(DatasetStatisticFilter, abc.ABC):
    """
    Ancestor for image segmentation statistics.
    """

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        raise NotImplementedError()

    def calculate(self, anns, preds, meta=None) -> Union[DatasetStatistic, DatasetStatisticList]:
        """
        Calculates the statistic from the tensors with annotations and predictions.

        :param anns: the list of annotations
        :param preds: the list of predictions
        :param meta: optional meta-data that is required for the calculation (domain-specific)
        :return: the generated statistic(s)
        :rtype: DatasetStatistic or DatasetStatisticList
        """
        raise NotImplementedError()


class MSCOCOImageSegmentationStatistic(ImageSegmentationStatistic, abc.ABC):
    """
    Ancestor for image segmentation statistics using MSCOCO.
    """

    def _prepare_data(self, data) -> Tuple[Optional[COCOAnnotations], Optional[COCOAnnotations], Optional[Dict[str, Any]]]:
        """
        Processes the image pairs and returns the tuple of MSCOCO annotations, MSCOCO predictions, class lookup and image lookup.

        :param data: the image pairs to use
        :type data: ImagePairList
        :return: the tuple of annotations, predictions, class lookup, image lookup
        :rtype: tuple
        """
        return prepare_mscoco_data(data, logger=self.logger())

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the statistic
        """
        result = None
        anns, preds, meta = self._prepare_data(data)
        if (anns is not None) and (preds is not None):
            result = self.calculate(preds, anns, meta=meta)

        return result
