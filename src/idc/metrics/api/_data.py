from dataclasses import dataclass
from typing import Any
from idc.api import ImageData


@dataclass
class MetricsDataPair:
    """
    Container for annotation/prediction pairs.
    """
    image_name: str = None
    annotation: ImageData = None
    prediction: ImageData = None

    def __str__(self):
        return self.image_name


@dataclass
class GlobalStatistic:
    """
    Container for a global statistic.
    """
    statistic: str = None
    value: Any = None


@dataclass
class ImageStatistic:
    """
    Container for a per-image statistic.
    """
    image_name: str = None
    statistic: str = None
    value: Any = None
