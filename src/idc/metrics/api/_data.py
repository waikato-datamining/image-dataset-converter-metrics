from dataclasses import dataclass
from idc.api import ImageData


@dataclass
class MetricsDataPair:
    """
    Simple container for annotation/prediction pairs.
    """
    image_name: str = None
    annotation: ImageData = None
    prediction: ImageData = None

    def __str__(self):
        return self.image_name
