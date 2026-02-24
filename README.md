# image-dataset-converter-metrics
[image-dataset-converter](https://github.com/waikato-datamining/image-dataset-converter) 
plugins for evaluating predictions against annotations.


## Installation

Via PyPI:

```bash
pip install image_dataset_converter_metrics
```

The latest code straight from the repository:

```bash
pip install git+https://github.com/waikato-datamining/image-dataset-converter-metrics.git
```

## Tools

### idc-metrics-help

```
usage: idc-metrics-help [-h] [-c [PACKAGE ...]] [-e EXCLUDED_CLASS_LISTERS]
                        [-T {stats-imgcls}] [-p NAME] [-f {text,markdown}]
                        [-L INT] [-o PATH] [-i FILE] [-t TITLE]
                        [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Tool for outputting help for plugins in various formats.

options:
  -h, --help            show this help message and exit
  -c [PACKAGE ...], --custom_class_listers [PACKAGE ...]
                        The custom class listers to use, uses the default ones
                        if not provided. (default: None)
  -e EXCLUDED_CLASS_LISTERS, --excluded_class_listers EXCLUDED_CLASS_LISTERS
                        The comma-separated list of class listers to exclude.
                        (default: None)
  -T {stats-imgcls}, --plugin_type {stats-imgcls}
                        The types of plugins to generate the help for.
                        (default: stats-imgcls)
  -p NAME, --plugin_name NAME
                        The name of the plugin to generate the help for,
                        generates it for all if not specified (default: None)
  -f {text,markdown}, --help_format {text,markdown}
                        The output format to generate (default: text)
  -L INT, --heading_level INT
                        The level to use for the heading (default: 1)
  -o PATH, --output PATH
                        The directory or file to store the help in; outputs it
                        to stdout if not supplied; if pointing to a directory,
                        automatically generates file name from plugin name and
                        help format (default: None)
  -i FILE, --index_file FILE
                        The file in the output directory to generate with an
                        overview of all plugins, grouped by type (in markdown
                        format, links them to the other generated files)
                        (default: None)
  -t TITLE, --index_title TITLE
                        The title to use in the index file (default: image-
                        dataset-converter statistics)
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
```


## Plugins

* [Pipeline](plugins/README.md)
* [Statistics](statistics/README.md)
