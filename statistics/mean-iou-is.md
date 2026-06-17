# mean-iou-is

* accepts: idc.metrics.api.ImagePairList
* generates: idc.metrics.statistic.DatasetStatistic, idc.metrics.statistic.DatasetStatisticList

Calculates the mean intersection over union (mIoU) for image segmentation data.

```
usage: mean-iou-is [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                   [-N LOGGER_NAME] [--skip] [-b] [-c]

Calculates the mean intersection over union (mIoU) for image segmentation
data.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -b, --exclude_background
                        Whether to exclude the background (= 1st layer) from
                        the calculation. (default: False)
  -c, --per_class       Whether to generate per class metrics. (default:
                        False)
```
