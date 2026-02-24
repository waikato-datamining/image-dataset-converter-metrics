# summary-statistics-ic

* accepts: idc.metrics.api.ImagePairList
* generates: idc.metrics.statistic.DatasetStatisticList

Calculates summary statistics for the incoming data pairs.

```
usage: summary-statistics-ic [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                             [-N LOGGER_NAME] [--skip] -s STATISTICS

Calculates summary statistics for the incoming data pairs.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -s STATISTICS, --statistics STATISTICS
                        The summary statistics to calculate. (default: None)
```
