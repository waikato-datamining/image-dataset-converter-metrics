from typing import List, Dict


def list_classes() -> Dict[str, List[str]]:
    return {
        "seppl.io.Reader": [
            "idc.metrics.reader",
        ],
        "seppl.io.Filter": [
            "idc.metrics.filter",
        ],
        "seppl.io.Writer": [
            "idc.metrics.writer",
        ],
    }
