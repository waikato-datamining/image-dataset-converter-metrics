# cohen-kappa-ic

* accepts: idc.metrics.api.ImagePairList
* generates: idc.metrics.statistic.DatasetStatistic

Calculates the Cohen-Kappa for image classification data.

```
usage: cohen-kappa-ic [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                      [-N LOGGER_NAME] [--skip] [-n NUM_CLASSES]

Calculates the Cohen-Kappa for image classification data.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -n NUM_CLASSES, --num_classes NUM_CLASSES
                        The number of classes in the dataset. (default: None)
```
