# load-metrics-pairs

* generates: idc.metrics.api.MetricsDataPair

Loads the annotation/prediction pairs using the respective sub-flows and forwards matching pairs for calculating metrics.

```
usage: load-metrics-pairs [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                          [-N LOGGER_NAME] [-a ANNOTATIONS_FLOW]
                          [-A {cmdline,file}] [-p PREDICTIONS_FLOW]
                          [-P {cmdline,file}]

Loads the annotation/prediction pairs using the respective sub-flows and
forwards matching pairs for calculating metrics.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -a ANNOTATIONS_FLOW, --annotations_flow ANNOTATIONS_FLOW
                        The subflow to loading the annotations (reader and
                        optional filter(s)). (default: None)
  -A {cmdline,file}, --annotations_flow_format {cmdline,file}
                        The format of the annotations pipeline. (default:
                        cmdline)
  -p PREDICTIONS_FLOW, --predictions_flow PREDICTIONS_FLOW
                        The subflow to loading the predictions (reader and
                        optional filter(s)). (default: None)
  -P {cmdline,file}, --predictions_flow_format {cmdline,file}
                        The format of the predictions pipeline. (default:
                        cmdline)
```
