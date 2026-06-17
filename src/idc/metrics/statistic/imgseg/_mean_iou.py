from ._image_segmentation_statistic import ImageSegmentationStatisticWithPerClass


class MeanIntersectionOverUnion(ImageSegmentationStatisticWithPerClass):
    """
    Calculates the mean intersection over union (mIoU) for image segmentation data.
    """

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "mean-iou-is"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Calculates the mean intersection over union (mIoU) for image segmentation data."

    def _statistic_name(self):
        """
        Returns the name for the statistic in the output.

        :return: the name
        :rtype: str
        """
        return "mIoU"

    def _initialize_statistic(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        import torchmetrics.segmentation
        self._statistic = torchmetrics.segmentation.MeanIoU(
            include_background=not self.exclude_background, per_class=self.per_class, input_format="one-hot")
