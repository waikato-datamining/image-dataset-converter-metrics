from typing import Dict
from seppl import Plugin
from idc.registry import REGISTRY


def available_imgcls_statistics() -> Dict[str, Plugin]:
    """
    Returns all image classification statistics plugins.
    """
    return REGISTRY.plugins("idc.metrics.statistic.imgcls.ClassificationStatistic")


def available_objdet_statistics() -> Dict[str, Plugin]:
    """
    Returns all image classification statistics plugins.
    """
    return REGISTRY.plugins("idc.metrics.statistic.objdet.ObjectDetectionStatistic")


def available_statistics() -> Dict[str, Plugin]:
    """
    Returns all image classification statistics plugins.
    """
    result = dict()
    result.update(available_imgcls_statistics())
    result.update(available_objdet_statistics())
    return result
