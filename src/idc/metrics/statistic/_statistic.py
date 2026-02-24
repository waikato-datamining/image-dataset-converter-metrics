import abc
from dataclasses import dataclass
from typing import List, Any

from seppl.io import BatchFilter
from idc.metrics.api import ImagePair, ImagePairList


@dataclass
class DatasetStatistic:
    """
    Container for a dataset statistic.
    """
    statistic: str = None
    value: Any = None

    def __str__(self):
        return "%s,%s" % (str(self.statistic), str(self.value))


class DatasetStatisticList(List[DatasetStatistic]):
    """
    Simple list of DatasetStatistic objects.
    """

    def _check_type(self, item):
        if not isinstance(item, DatasetStatistic):
            raise Exception("Only accepts objects of type: %s" % str(type(DatasetStatistic)))

    def append(self, item):
        self._check_type(item)
        super().append(item)

    def extend(self, iterable):
        for item in iterable:
            self._check_type(item)
        super().extend(iterable)

    def insert(self, index, object):
        self._check_type(object)
        super().insert(index, object)

    def __str__(self):
        return str(self)


@dataclass
class ImageStatistic:
    """
    Container for a per-image statistic.
    """
    image_name: str = None
    statistic: str = None
    value: Any = None

    def __str__(self):
        return "%s,%s,%s" % (str(self.image_name), str(self.statistic), str(self.value))


class ImageStatisticList(List[ImageStatistic]):
    """
    Simple list of ImageStatistic objects.
    """

    def _check_type(self, item):
        if not isinstance(item, ImageStatistic):
            raise Exception("Only accepts objects of type: %s" % str(type(ImageStatistic)))

    def append(self, item):
        self._check_type(item)
        super().append(item)

    def extend(self, iterable):
        for item in iterable:
            self._check_type(item)
        super().extend(iterable)

    def insert(self, index, object):
        self._check_type(object)
        super().insert(index, object)

    def __str__(self):
        return str(self)


class DatasetStatisticFilter(BatchFilter, abc.ABC):
    """
    Base class for filters that calculate global statistics. Processes data in batches.
    """

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
        return [DatasetStatistic]

    def calculate(self, anns, preds) -> DatasetStatistic:
        """
        Calculates the statistic from the tensors with annotations and predictions.

        :param anns: the tensor with the class label indices of the annotations
        :param preds: the tensor with the class label indices of the predictions
        :return: the generated statistic
        :rtype: DatasetStatistic
        """
        raise NotImplementedError()


class ImageStatisticFilter(BatchFilter, abc.ABC):
    """
    Base class for per-image statistics.
    """

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImagePair]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageStatistic]

    def calculate(self, anns, preds) -> ImageStatistic:
        """
        Calculates the statistic from the tensors with annotations and predictions.

        :param anns: the tensor with the class label indices of the annotations
        :param preds: the tensor with the class label indices of the predictions
        :return: the generated statistic
        :rtype: ImageStatistic
        """
        raise NotImplementedError()
