# mean-average-precision-od

* accepts: idc.metrics.api.ImagePairList
* generates: idc.metrics.statistic.DatasetStatistic, idc.metrics.statistic.DatasetStatisticList

Calculates the mean average precision (mAP) for object detection data. Retrieves the 'score' meta-data value from the object detection predictions, uses 1.0 if not present.

```
usage: mean-average-precision-od [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                 [-N LOGGER_NAME] [--skip] [-c]
                                 [-a {micro,macro}]
                                 [-b {pycocotools,faster-coco-eval}]

Calculates the mean average precision (mAP) for object detection data.
Retrieves the 'score' meta-data value from the object detection predictions,
uses 1.0 if not present.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -c, --class_metrics   Whether to generate class-level metrics. (default:
                        False)
  -a {micro,macro}, --average {micro,macro}
                        The type of average to compute. (default: micro)
  -b {pycocotools,faster-coco-eval}, --backend {pycocotools,faster-coco-eval}
                        The type of backend to use for computation. (default:
                        pycocotools)
```
