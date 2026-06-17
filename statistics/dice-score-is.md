# dice-score-is

* accepts: idc.metrics.api.ImagePairList
* generates: idc.metrics.statistic.DatasetStatistic, idc.metrics.statistic.DatasetStatisticList

Calculates the dice score for image segmentation data.

```
usage: dice-score-is [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [-N LOGGER_NAME] [--skip] [-b] [-n NUM_CLASSES]
                     [-a {micro,macro,weighted,none}] [-g {samplewise,global}]

Calculates the dice score for image segmentation data.

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
  -n NUM_CLASSES, --num_classes NUM_CLASSES
                        The number of classes in the dataset. (default: None)
  -a {micro,macro,weighted,none}, --average {micro,macro,weighted,none}
                        The type of average to compute. (default: macro)
  -g {samplewise,global}, --aggregation_level {samplewise,global}
                        The aggregation level to use. (default: samplewise)
```
