from typing import Union

from sklearn.metrics import cohen_kappa_score

from ._classification_statistic import ClassificationStatistic
from idc.metrics.statistic import DatasetStatistic, DatasetStatisticList


class CohenKappa(ClassificationStatistic):
    """
    Calculates the Recall for image classification data.
    """

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "cohen-kappa-ic"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the Cohen-Kappa for image classification data."

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "Cohen-Kappa"

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
        result.value = cohen_kappa_score(anns, preds)
        return result
