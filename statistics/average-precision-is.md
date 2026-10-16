# average-precision-is

* accepts: idc.metrics.api.ImagePairList
* generates: idc.metrics.statistic.DatasetStatistic, idc.metrics.statistic.DatasetStatisticList

Calculates the selected average precision (AP) statistics for image segmentation data. Uses 1.0 for the score.

```
usage: average-precision-is [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                            [-N LOGGER_NAME] [--skip] -s
                            {AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl}
                            [{AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl} ...]
                            [-t {bbox,segm}]

Calculates the selected average precision (AP) statistics for image
segmentation data. Uses 1.0 for the score.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -s {AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl} [{AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl} ...], --statistic {AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl} [{AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl} ...]
                        The statistic(s) to compute. (default: None)
  -t {bbox,segm}, --iou_type {bbox,segm}
                        How to evaluate the intersect-over-union (IoU).
                        (default: segm)
```
