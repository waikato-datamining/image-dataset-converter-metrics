from typing import Union, List

from sklearn.metrics import precision_score

from ._classification_statistic import ClassificationStatisticWithAverage
from idc.metrics.statistic import DatasetStatistic, DatasetStatisticList


class Precision(ClassificationStatisticWithAverage):
    """
    Calculates the Precision for image classification data.
    """

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "precision-ic"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the Precision for image classification data."

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "Precision"

    def _averages(self) -> List[str]:
        """
        Returns the possible averages.

        :return: the averages
        :rtype: list
        """
        return ["micro", "macro", "samples", "weighted", "binary"]

    def _default_average(self) -> str:
        """
        Returns the default average.

        :return: the default
        :rtype: str
        """
        return "binary"

    def calculate(self, anns, preds, meta=None) -> Union[DatasetStatistic, DatasetStatisticList]:
        """
        Calculates the statistic from the tensors with annotations and predictions.

        :param anns: the tensor with the class label indices of the annotations
        :param preds: the tensor with the class label indices of the predictions
        :param meta: optional meta-data that is required for the calculation (domain-specific)
        :return: the generated statistic(s)
        :rtype: DatasetStatistic or DatasetStatisticList
        """
        result = DatasetStatistic(statistic=self._statistic_name())
        result.value = precision_score(anns, preds, average=self.average)
        return result
