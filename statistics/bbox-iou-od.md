# bbox-iou-od

* accepts: idc.metrics.api.ImagePairList
* generates: idc.metrics.statistic.DatasetStatistic, idc.metrics.statistic.DatasetStatisticList

Calculates the Intersection over Union (= Jaccard Index) for object detection bounding boxes.

```
usage: bbox-iou-od [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                   [-N LOGGER_NAME] [--skip] [-c] [-t IOU_THRESHOLD] [-i]

Calculates the Intersection over Union (= Jaccard Index) for object detection
bounding boxes.

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
  -t IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
                        The threshold to apply. (default: None)
  -i, --ignore_labels   When set, will ignore labels during IoU computation.
                        (default: False)
```
