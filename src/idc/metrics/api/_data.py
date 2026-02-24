from dataclasses import dataclass
from typing import List
from idc.api import ImageData


@dataclass
class ImagePair:
    """
    Container for annotation/prediction pairs.
    """
    image_name: str = None
    annotation: ImageData = None
    prediction: ImageData = None

    def __str__(self):
        return self.image_name


class ImagePairList(List[ImagePair]):
    """
    Simple list of ImagePair objects.
    """

    def _check_type(self, item):
        if not isinstance(item, ImagePair):
            raise Exception("Only accepts objects of type: %s" % str(type(ImagePair)))

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
