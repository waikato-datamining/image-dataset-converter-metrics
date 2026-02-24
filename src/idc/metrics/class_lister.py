from typing import List, Dict


def list_classes() -> Dict[str, List[str]]:
    return {
        "seppl.io.Reader": [
            "idc.metrics.reader",
        ],
        "seppl.io.Filter": [
            "idc.metrics.filter",
            "idc.metrics.filter.depth",
            "idc.metrics.filter.imgcls",
            "idc.metrics.filter.imgseg",
            "idc.metrics.filter.objdet",
        ],
        "seppl.io.Writer": [
            "idc.metrics.writer",
            "idc.metrics.writer.imgcls",
        ],
        "idc.metrics.statistic.imgcls.ClassificationStatistic": [
            "idc.metrics.statistic.imgcls",
        ],
    }
