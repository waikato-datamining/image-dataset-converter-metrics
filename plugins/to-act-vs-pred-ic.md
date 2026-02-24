# to-act-vs-pred-ic

* accepts: idc.metrics.api.ImagePair

Outputs a CSV file with actual vs predicted columns.

```
usage: to-act-vs-pred-ic [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                         [-N LOGGER_NAME] [--skip] -o OUTPUT [-i COLUMN_IMAGE]
                         [-a COLUMN_ACTUAL] [-p COLUMN_PREDICTED]

Outputs a CSV file with actual vs predicted columns.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -o OUTPUT, --output OUTPUT
                        The CSV file to store the actual vs predicted data in.
                        Supported placeholders: {HOME}, {CWD}, {TMP} (default:
                        None)
  -i COLUMN_IMAGE, --column_image COLUMN_IMAGE
                        The column name for the image name. (default: Image)
  -a COLUMN_ACTUAL, --column_actual COLUMN_ACTUAL
                        The column name for the actual values. (default:
                        Actual)
  -p COLUMN_PREDICTED, --column_predicted COLUMN_PREDICTED
                        The column name for the predicted values. (default:
                        Predicted)
```

Available placeholders:

* `{HOME}`: The home directory of the current user.
* `{CWD}`: The current working directory.
* `{TMP}`: The temp directory.
