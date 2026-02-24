from ._classification_statistic import ClassificationStatistic


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

    def _initialize_statistic(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        import torchmetrics
        self._statistic = torchmetrics.CohenKappa(task="multiclass", num_classes=self.num_classes)
